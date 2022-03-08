import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 2.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):

    race_encoder =  {'afroamerican' : 0 , 'asian' : 1, 'caucasian' : 2}
    gender_encoder = {'male': 0, 'female': 1}
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0, load_residuals=False, ignore_corrupted= True):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        extended_labels_path = Path(data_dir).joinpath(f"allcategories_{data_type}.csv")
        residuals_path = Path(data_dir).joinpath(f"residuals_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.load_residuals = load_residuals
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        self.gender = []
        self.race = []
        self.residual = []

        df = pd.read_csv(str(csv_path))
        df_extended = pd.read_csv(str(extended_labels_path), index_col=0)
        if load_residuals:
            df_residuals = pd.read_csv(str(residuals_path), index_col=0)

        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]
            row_extended = df_extended.loc[img_name]

            if img_name in ignore_img_names and ignore_corrupted:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])
            self.gender.append(row_extended['gender'])
            self.race.append(row_extended['race'])
            if load_residuals:
                row_residuals = df_residuals.loc[img_name]
                self.residual.append((row_residuals["residual"]))

       


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # set the augmentation for half imgs  
        #self.augment = idx%2 == 0
        img_path = self.x[idx]
        age = self.y[idx]
        gender = self.gender_encoder[self.gender[idx]] 
        race = self.race_encoder[self.race[idx]] 

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        x = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        if self.load_residuals:
            residual = self.residual[idx]
            return x, residual, gender, race
        else :
            return x, np.clip(round(age), 0, 100), gender, race



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDataset(args.data_dir, "train")
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "valid")
    print("valid dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "test")
    print("test dataset len: {}".format(len(dataset)))


if __name__ == '__main__':
    main()
