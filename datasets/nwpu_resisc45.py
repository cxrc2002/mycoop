import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "airplane": "airplane",
    "baseball_diamond": "baseball_diamond",
    "beach": "beach",
    "bridge": "bridge",
    "chaparral": "chaparral",
    "church": "church",
    "cloud": "cloud",
    "desert": "desert",
    "freeway": "freeway",
    "golf_course": "golf_course",
    "harbor": "harbor",
    "island": "island",
    "lake": "lake",
    "meadow": "meadow",
    "mobile_home_park": "mobile_home_park",
    "mountain": "mountain",
    "palace": "palace",
    "railway": "railway",
    "rectangular_farmland": "rectangular_farmland",
    "roundabout": "roundabout",
    "sea_ice": "sea_ice",
    "ship": "ship",
    "sparse_residential": "sparse_residential",
    "stadium": "stadium",
    "wetland": "wetland",
    "commercial_area": "commercial_area",
    "industrial_area": "industrial_area",
    "overpass": "overpass",
    "railway_station": "railway_station",
    "runway": "runway",
    "snowberg": "snowberg",
    "storage_tank": "storage_tank",
    "tennis_court": "tennis_court",
    "terrace": "terrace",
    "thermal_power_station": "thermal_power_station",
    "airport": "airport",
    "basketball_court": "basketball_court",
    "circular_farmland": "circular_farmland",
    "dense_residential": "dense_residential",
    "forest": "forest",
    "ground_track_field": "ground_track_field",
    "intersection": "intersection",
    "medium_residential": "medium_residential",
    "parking_lot": "parking_lot",
    "river": "river"
}


@DATASET_REGISTRY.register()
class NWPU_RESISC45(DatasetBase):
    dataset_dir = "nwpu_resisc45"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image")
        self.split_path = os.path.join(self.dataset_dir, "split_nwpu_resisc45.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
