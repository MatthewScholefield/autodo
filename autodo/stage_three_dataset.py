from collections import namedtuple

import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import random
from os.path import join
from prettyparse import Usage
from torch.utils.data.dataset import Dataset as TorchDataset

from autodo.dataset import Dataset

k = np.array([[2304.5479, 0, 1686.2379],
              [0, 2305.8757, 1354.9849],
              [0, 0, 1]], dtype=np.float32)
IMG_WIDTH = 448
IMG_HEIGHT = 128

StageThreeLabel = namedtuple('StageThreeLabel', 'x y z')


def _load_centers(df, image_id):
    return df[df['image_id'] == image_id][['center_x', 'center_y']].values.T


class StageThreeDataset(TorchDataset):
    usage = Usage('''
        :stage_two_csv str
            Csv of stage two output

        :dataset_folder str
            Folder to load images from
    ''')

    @classmethod
    def from_args(cls, args, train=True):
        return cls(Dataset.from_folder(args.dataset_folder), args.stage_two_csv, train)

    def __init__(self, auto_dataset: Dataset, stage_two_csv, train=True):
        super().__init__()
        self.auto_dataset = auto_dataset
        self.df = pd.read_csv(stage_two_csv)
        self.datas = list(self.df.index)
        random.seed(1234)
        random.shuffle(self.datas)
        cutoff = int(len(self.datas) * 0.7)
        if train:
            self.datas = self.datas[:cutoff]
        else:
            self.datas = self.datas[cutoff:]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        row = self.df.iloc[self.datas[idx]]
        image_id = row['image_id']
        xcenters, ycenters = _load_centers(self.df, image_id)
        input_tensor = self.make_input(join(self.auto_dataset.images_folder[0], image_id + '.jpg'), xcenters, ycenters)
        output, out_mask = self.make_target(image_id)
        return input_tensor, output, out_mask

    @staticmethod
    def make_input(filename, xcenters, ycenters, shape_x=IMG_WIDTH, shape_y=IMG_HEIGHT, cutoff_y=1600):
        width = 3384
        height = 2710
        padding_x = width // 6
        img = np.zeros((height, width + padding_x * 2, 3), dtype='float32')
        img[:, padding_x:-padding_x, :] = plt.imread(filename).astype('float32') / 255
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype='float32')
        output = np.zeros((img.shape[0], img.shape[1], 3), dtype='float32')
        input_tensor = np.concatenate([img, mask], axis=2)
        input_tensor, output = input_tensor[cutoff_y:], output[cutoff_y:]
        input_tensor = cv2.resize(input_tensor, (shape_x, shape_y))
        for xcenter, ycenter in zip(xcenters, ycenters):
            xcenter = xcenter * width
            ycenter = ycenter * height
            if xcenter < -padding_x or xcenter > width + padding_x or ycenter < 0 or ycenter > height:
                # print('fuck')  # if result is too ridiculous
                continue
            xcenter = (xcenter + padding_x) / (width + padding_x * 2) * shape_x
            ycenter = (ycenter - cutoff_y) / (height - cutoff_y) * shape_y
            input_tensor[int(ycenter), int(xcenter), 3] = 1
        return input_tensor.astype('float32').transpose((2, 0, 1))

    def make_target(self, image_id, shape_y=IMG_HEIGHT, shape_x=IMG_WIDTH, cutoff_y=1600):
        width = 3384
        height = 2710
        padding_x = width // 6
        img = np.zeros((height, width + padding_x * 2, 3), dtype='float32')
        filename = join(self.auto_dataset.images_folder[0], image_id + '.jpg')
        img[:, padding_x:-padding_x, :] = plt.imread(filename).astype('float32') / 255
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype='float32')
        output = np.zeros((img.shape[0], img.shape[1], 3), dtype='float32')
        df = self.df[self.df['image_id'] == image_id]

        xcenters, ycenters = df[['center_x', 'center_y']].values.T
        labels = self.auto_dataset.labels_s[image_id]
        xs, ys, zs = np.array([
            [label.x, label.y, label.z]
            for label in labels
        ]).T

        output = output[cutoff_y:]
        target = cv2.resize(output, (shape_x, shape_y))

        input_tensor = np.concatenate([img, mask], axis=2)
        input_tensor = input_tensor[cutoff_y:]
        input_tensor = cv2.resize(input_tensor, (shape_x, shape_y))

        out_mask = cv2.resize(input_tensor[:, :, 3], (shape_x, shape_y))[:, :, np.newaxis]
        for xcenter, ycenter, x, y, z in zip(xcenters, ycenters, xs, ys, zs):
            if xcenter < -padding_x or xcenter > width + padding_x or ycenter < 0 or ycenter > height:
                # print('fuck')  # if result is too ridiculous
                continue
            xcenter = (xcenter + padding_x) / (width + padding_x * 2) * shape_x
            ycenter = (ycenter - cutoff_y) / (height - cutoff_y) * shape_y
            target[int(ycenter), int(xcenter)] = np.array([x, y, z]) / 100
            out_mask[int(ycenter), int(xcenter)] = 1
        return target.astype('float32').transpose((2, 0, 1)), out_mask.astype('float32').transpose((2, 0, 1))

    @staticmethod
    def decode_output(network_output, xcenters, ycenters):
        print(network_output.shape)
        return [
            StageThreeLabel(*map(float, network_output.cpu().detach().numpy().transpose((1, 2, 0))[int(xcenter), int(ycenter)] * 100))
            for xcenter, ycenter in zip(xcenters, ycenters)
        ]
