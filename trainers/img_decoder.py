import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.optim as optim
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def load_clip_to_cpu():
    backbone_name = "RN50"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # remote_weights_path = "./checkpoints/RemoteCLIP-RN50.pt"
    # remote_state_dict = torch.load(remote_weights_path, map_location="cpu")
    # model = clip.build_model(remote_state_dict)
    # message = model.load_state_dict(remote_state_dict)
    # print(message)
    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        #self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts):
        x = prompts
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 17] @ self.text_projection
        return x



class ImgDecoder(nn.Module):
    def __init__(self, n_ctx):
        super().__init__()
        # 超参数
        self.d_model = 512
        self.nhead = 8
        self.num_decoder_layers = 12
        self.n_ctx = n_ctx
        self.back_project = nn.Linear(1024, 512)
        # 定义解码器层
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers)

    def forward(self, input_memory):
        x = self.back_project(input_memory)  # (n, 1, 512)
        batch_size = input_memory.size(0)

        # init decoder_input
        decoder_input = torch.zeros(batch_size, 1, self.d_model, device=input_memory.device)

        # final output
        output_sequence = []

        # 自回归生成过程
        for i in range(self.n_ctx):
            decoder_output = self.transformer_decoder(decoder_input, x)
            # 将新生成的时间步添加到输出序列
            output_sequence.append(decoder_output)
            # 更新解码器输入为最新生成的时间步
            if i < self.n_ctx - 1:  # 如果不是最后一个时间步
                decoder_input = torch.cat([decoder_input, decoder_output], dim=1)  # (batchsize, i+2, d_model)
            else:
                break  # 最后一个时间步不需要更新输入
        # 合并所有时间步的输出
        final_output = torch.cat(output_sequence, dim=1)  # (batchsize, n_ctx, d_model)

        return final_output


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        # 最大化余弦相似度（最小化 1 - 余弦相似度）
        return 1 - self.cos_sim(pred, target).mean()



n_ctx = 16
batch_size = 10
clip_model = load_clip_to_cpu()
clip_model.float()
print("clip_model created successfully")
textencoder = TextEncoder(clip_model)
print("textencoder created successfully")

orgctx = torch.randn(batch_size, 16, 512).to(clip_model.dtype)
#print(orgctx)

prompt_prefix = " ".join(["X"] * n_ctx)
prompt_prefix = prompt_prefix + "."
tokenized_prefix = clip.tokenize(prompt_prefix)
embedding = clip_model.token_embedding(tokenized_prefix).type(clip_model.dtype)     #(1,77,512)
embedding = embedding.expand(batch_size,-1,-1)   #(100,77,512)
prompts = torch.cat(
                [
                    embedding[:, :1, :],  # (n, n_cls, 1, dim)
                    orgctx,     # (n, n_cls, n_ctx, dim)
                    embedding[:, 17:, :],  # (n, n_cls, *, dim)
                ],
                dim=1,
            )

#print(prompts)
encoded_text_features = textencoder(prompts)                  #100,1024
encoded_text_features = encoded_text_features.unsqueeze(1)     #100,1,1024
print("encoded_text_features shape:",encoded_text_features.shape)

imgdecoder = ImgDecoder(n_ctx)
print("imgdecoder created successfully")
criterion = CosineSimilarityLoss()
optimizer = optim.Adam(imgdecoder.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    print(f"epoch[{epoch}]:")
    imgdecoder.train()

    # Forward
    decoded_output = imgdecoder(encoded_text_features)  # (batchsize, n_ctx, d_dim)

    # 计算损失
    loss = criterion(decoded_output, orgctx)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")