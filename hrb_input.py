from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.parameters import *
import random
import csv

mean = [0.729, 0.686, 0.577]
std = [0.195, 0.205, 0.229]

rescale = transforms.Resize((HEIGHT, WIDTH))
normalize = transforms.Normalize(mean, std)

preprocess = transforms.Compose([rescale, transforms.ToTensor(), normalize])

v_flip = transforms.RandomVerticalFlip()
h_flip = transforms.RandomHorizontalFlip()
s_color = transforms.ColorJitter(saturation=0.1)
c_color = transforms.ColorJitter(contrast=0.1)
h_color = transforms.ColorJitter(hue=0.1)
grayscale = transforms.RandomGrayscale()

augment1 = transforms.RandomApply(
    [h_flip, s_color, c_color, h_color, v_flip, grayscale])
augment2 = transforms.RandomChoice(
    [h_flip, s_color, c_color, h_color, v_flip, grayscale])
augment3 = transforms.RandomOrder(
    [h_flip, s_color, c_color, h_color, v_flip, grayscale])


class TrainDataset(Dataset):
    def __init__(self, transform=None):

        with open(TRAIN, 'r') as f:
            self.train_paths = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, idx):
        csv_file = self.train_paths[idx]

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            row = next(reader)

        img_path = IMG_DIR + row[0]
        pil_img = Image.open(img_path)

        chance = random.random()

        if chance < P1:
            pil_img = augment2(pil_img)
        elif chance < P2:
            pil_img = augment3(pil_img)
        elif chance < P3:
            pil_img = augment1(pil_img)

        ten_img = preprocess(pil_img)
        label = int(row[-1])

        return {'image': ten_img, 'label': label}


class ValidationDataset(Dataset):
    def __init__(self, transform=None):

        with open(VAL, 'r') as f:
            self.validation_paths = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.validation_paths)

    def __getitem__(self, idx):
        csv_file = self.validation_paths[idx]

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            row = next(reader)

        img_path = IMG_DIR + row[0]
        pil_img = Image.open(img_path)
        ten_img = preprocess(pil_img)
        label = int(row[-1])

        return {'image': ten_img, 'label': label}


class TestDataset(Dataset):
    def __init__(self, transform=None):

        with open(TEST, 'r') as f:
            self.test_paths = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.test_paths)

    def __getitem__(self, idx):
        csv_file = self.test_paths[idx]

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            row = next(reader)

        img_path = IMG_DIR + row[0]
        pil_img = Image.open(img_path)
        ten_img = preprocess(pil_img)
        label = int(row[-1])

        return {'image': ten_img, 'label': label}


if __name__ == "__main__":
    dataset = TestDataset()
    data = dataset.__getitem__(1000)

    print(data['image'].size(), data['label'])
