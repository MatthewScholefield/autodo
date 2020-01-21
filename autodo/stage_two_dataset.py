from collections import namedtuple
from math import cos, sin, asin

import cv2
import numpy as np
import random
from os.path import join
from prettyparse import Usage
from torch.utils.data import Dataset as TorchDataset

from autodo.dataset import Dataset

StageTwoLabel = namedtuple('StageTwoLabel', 'yaw pitch roll center_x center_y')


class StageTwoDataset(TorchDataset):
    usage = Usage('''
        :cropped_folder str
            Folder of cropped images

        :dataset_folder str
            Dataset folder
        
        :-nd --network-data str -
            Network output csv
    ''')

    @classmethod
    def from_args(cls, args, train=True):
        return cls(args.cropped_folder, args.dataset_folder, args.network_data, train)

    def __init__(self, cropped_folder, dataset_folder, network_data_file=None, train=True):
        super().__init__()
        self.cropped_folder = cropped_folder
        self.dataset = Dataset.from_folder(dataset_folder)
        if network_data_file:
            import pandas
            df = pandas.read_csv(network_data_file)
            self.network_data = {}
            for a, b in df.groupby('image_id'):
                self.network_data[a] = b
        else:
            self.network_data = None
        self.datas = []
        for image_id, boxes in self.dataset.labels_s.items():
            if self.network_data:
                if image_id not in self.network_data:
                    continue
                data = self.network_data[image_id]
            else:
                data = None
            self.datas.extend([
                (image_id, box_id)
                for box_id in range(len(boxes))
                if data is None or box_id < len(data)
            ])
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
        image_id, box_id = self.datas[idx]
        label = self.dataset.labels_s[image_id][box_id]
        if self.network_data:
            row = self.network_data[image_id].iloc[box_id]
            box = [row[i] for i in ['xmin', 'ymin', 'xmax', 'ymax']]
        else:
            box = self.dataset.calc_bbox(label)

        xmin, ymin, xmax, ymax = box
        pred_filename = join(self.cropped_folder, image_id + '-{:02}.jpg'.format(box_id))
        verts = self.dataset.project_verts(np.array([[0, 0, 0]]), label)
        xcenter, ycenter = verts[0]

        xmiddle = (xmax + xmin) / 2
        ymiddle = (ymax + ymin) / 2
        xcenter = (xcenter - xmiddle) / (max(xmax - xmin, ymax - ymin) / 2)
        ycenter = (ycenter - ymiddle) / (max(ymax - ymin, ymax - ymin) / 2)

        return self.load_image(pred_filename, box), (np.array([
            cos(label.yaw) > 0,
            sin(label.yaw) * ((cos(label.yaw) > 0) * 2 - 1),
            label.pitch,
            rotate(label.roll, np.pi),
            xcenter,
            ycenter
        ])).astype('float32')

    @staticmethod
    def load_image(image_file, box):
        import matplotlib.pylab as plt
        img = plt.imread(image_file).astype('float32') / 255
        height, width, channels = img.shape
        mesh = get_numpy_mesh(height, width, box)
        img = np.concatenate([img, mesh], axis=2)
        img = pad_img(img, size=(112, 112, 5))
        img = img.transpose((2, 0, 1))
        return img.astype('float32')

    @staticmethod
    def decode_label(network_output):
        cos_dir, sin_val, pitch, roll_val, center_x, center_y = map(float, network_output)
        sin_val = max(-1.0, min(1.0, sin_val))
        return StageTwoLabel(
            asin(sin_val * ((cos_dir > 0.5) * 2 - 1)),
            pitch,
            rotate(roll_val, -np.pi),
            center_x,
            center_y
        )


def get_numpy_mesh(shape_y, shape_x, box):
    mesh = np.zeros([shape_y, shape_x, 2])

    xmin, ymin, xmax, ymax = box
    mg_y, mg_x = np.meshgrid(np.linspace(ymin, ymax, shape_y), np.linspace(xmin, xmax, shape_x), indexing='ij')
    mesh[:, :, 0] = mg_y
    mesh[:, :, 1] = mg_x
    return mesh.astype('float32')


def pad_img(img, size=(112, 112, 3)):
    padded_img = np.zeros(size)
    pad_center_y = size[0] / 2
    pad_center_x = size[1] / 2
    if img.shape[0] > img.shape[1]:
        newwidth = int(img.shape[1] / img.shape[0] * size[0] // 2 * 2)
        img = cv2.resize(img, (newwidth, size[0]))
        padded_img[:, int(pad_center_x - newwidth // 2):int(pad_center_x + newwidth // 2)] = img
    else:
        newheight = int(img.shape[0] / img.shape[1] * size[1] // 2 * 2)
        img = cv2.resize(img, (size[1], newheight))
        padded_img[int(pad_center_y - newheight // 2):int(pad_center_y + newheight // 2), :] = img
    return padded_img


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x
