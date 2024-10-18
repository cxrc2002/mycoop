import os.path as osp

from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)


        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)    #list
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        #self.ctx = ctx_vectors
        #print(self.ctx.shape)  #(n_ctx,512)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]   # lsit len(prompts)=10=len(classnames)  ["xxxx class1","xxxxx class2",....]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])    # get token/word id
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS size[batch_size, 1, embedding_dim]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION   # end middle start
        #self.w = nn.Parameter(torch.tensor(0.5, dtype=clip_model.dtype))

    def forward(self, img_prompts):
        batch_size = img_prompts.shape[0]
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(batch_size, -1, -1)
            ctx = ctx + img_prompts              # (n, n_ctx, dim)
            #print(ctx.shape)
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            ctx = ctx.transpose(dim0=0, dim1=1)      # (batchsize, n_cls ,n_ctx, dim)
            # print(ctx.shape)             
        prefix = self.token_prefix
        suffix = self.token_suffix
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n, n_cls, 1, dim)
                    ctx,     # (n, n_cls, n_ctx, dim)
                    suffix,  # (n, n_cls, *, dim)
                ],
                dim=2,
            )
            # (n, n_cls, 77, dim)
        return prompts


class Image2Prompts(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_ctx = cfg.TRAINER.COOP.N_CTX
        # n_ctx = n_ctx // 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_ctx, kernel_size=1).to(clip_model.dtype)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=n_ctx, out_channels=n_ctx, kernel_size=1).to(clip_model.dtype)

        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.pool = nn.AdaptiveAvgPool1d(self.ctx_dim).to(clip_model.dtype)

    def forward(self, image_features):
        # img_size(batchsize, 1024)
        image_features = self.pool(image_features)
        image_features = image_features.unsqueeze(1)

        # 应用一维卷积
        output_features = self.conv1(image_features)
        output_features = self.relu(output_features)
        output_features = self.conv2(output_features)
        #output_features = self.linear(output_features)
        # 输出的形状是 (batch_size, n_ctx, ctx_dim)
        #print(output_features.shape)

        return output_features


class ImgDecoder(nn.Module):
    def __init__(self, n_ctx, clip_model):
        super().__init__()

        self.dtype = clip_model.dtype
        self.d_model = 512
        self.nhead = 8
        self.num_decoder_layers = 12
        self.n_ctx = n_ctx
        
        self.back_project = nn.Linear(1024, 512).to(self.dtype)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nhead).to(self.dtype)
        self.transformer_decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers).to(self.dtype)

    def forward(self, img_features):
        
        img_features = img_features.type(self.dtype)
        img_features = img_features.unsqueeze(1)
        x = self.back_project(img_features)  # (n, 1, 512)
        batch_size = img_features.size(0)    # n

        # init decoder_input
        decoder_input = torch.zeros(batch_size, 1, self.d_model, device=img_features.device).type(self.dtype)  # (n,1,512)

        # final output
        output_sequence = []

        # predict n_ctx
        for i in range(self.n_ctx):
            
            decoder_output = self.transformer_decoder(decoder_input, x)
            output_sequence.append(decoder_output)

            if i < self.n_ctx - 1:  # if not the last time
                # decoder_input = torch.cat([decoder_input, decoder_output], dim=1)  # (batchsize, i+2, d_model)
                decoder_input = decoder_output

            else:
                break  

        final_output = torch.cat(output_sequence, dim=1)  # (batchsize, n_ctx, d_model)

        return final_output






class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.img2prompts = Image2Prompts(cfg, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.imgdecoder = ImgDecoder(cfg.TRAINER.COOP.N_CTX,clip_model)
        #self.imgdecoder.load_state_dict(torch.load("./checkpoints/imgdecoder_nctx8_batchsize50_ep100.pth"))
        #print("succcessfully")

    def forward(self, image):
        
        image_features = self.image_encoder(image.type(self.dtype))     # (n, 1024)
        img_prompts = self.imgdecoder(image_features)

        #img_prompts = self.img2prompts(image_features)
        prompts = self.prompt_learner(img_prompts)
        tokenized_prompts = self.tokenized_prompts
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = []
        for i in range(prompts.shape[0]):
            meta_prompts = prompts[i]
            #print(meta_prompts.shape)
            meta_features = self.text_encoder(meta_prompts, tokenized_prompts)
            #print(meta_features.shape)
            meta_features = meta_features / meta_features.norm(dim=-1, keepdim=True)
            #text_features.append(meta_features)
            logit_scale = self.logit_scale.exp()
            score = logit_scale*image_features[i] @ meta_features.t()
            #print(score.shape)
            logits.append(score)
            #print(logits)

        logits = torch.stack(logits, dim=0)
        #print((logits.requires_grad))
        #print(logits.shape)
        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "imgdecoder" not in name:
                param.requires_grad_(False)
            # if "img2prompts" not in name:
            #     param.requires_grad_(False)
            # else:
                # print(name)
                # print(param.requires_grad)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)   #param
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("img2prompts", self.model.img2prompts, self.optim, self.sched)
        self.register_model("imgdecoder",self.model.imgdecoder,self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
