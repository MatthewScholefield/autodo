from collections import namedtuple
from itertools import zip_longest

import csv
import json
import numpy as np
from glob import glob
from lazy import lazy
from os.path import join

from autodo.car_models import car_id2name
from autodo.utils import euler_to_rot

CarLabel = namedtuple('CarLabel', 'car_id yaw pitch roll x y z')
CarData = namedtuple('CarData', 'car_type faces vertices')
num_params = 7


class Dataset:
    camera_intrinsic = np.array([
        [2304.5479, 0, 1686.2379],
        [0, 2305.8757, 1354.9849],
        [0, 0, 1]
    ], dtype=np.float32)

    def __init__(self, csv_file, train_images_folder, test_images_folder, car_json_folder):
        self.csv_file = csv_file
        self.images_folder = (train_images_folder, test_images_folder)
        self.car_json_folder = car_json_folder

    def get_train_file_names(self):
        return glob(join(self.images_folder[0], '*.jpg'))

    def get_test_file_names(self):
        return glob(join(self.images_folder[1], '*.jpg'))

    def calc_verts(self, label: CarLabel):
        vertices = self.car_models[label.car_id].vertices
        yaw, pitch, roll = -label.pitch, -label.yaw, -label.roll
        mat_rot = np.eye(4)
        mat_rot[:3, 3] = np.array([label.x, label.y, label.z])
        mat_rot[:3, :3] = euler_to_rot(yaw, pitch, roll).T
        mat_rot = mat_rot[:3, :]
        vertices_4 = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        vertices_4[:, :-1] = vertices
        vertices_4 = vertices_4.T
        img_cor_points = np.dot(self.camera_intrinsic, np.dot(mat_rot, vertices_4))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        width = 3384
        height = 2710
        pix_pos = img_cor_points[:, :2]
        pix_pos[:, 0] /= width
        pix_pos[:, 1] /= height
        return pix_pos

    def calc_bbox(self, label: CarLabel):
        pix_pos = self.calc_verts(label)
        left_r, top_r = pix_pos.min(axis=0)
        right_r, bot_r = pix_pos.max(axis=0)
        return np.array([left_r, top_r, right_r, bot_r])

    @lazy
    def labels_s(self):
        return dict(self._parse_labels_s(self.csv_file))

    @lazy
    def car_models(self) -> dict:
        return {car_id: self._load_car(label.name) for car_id, label in car_id2name.items()}

    @classmethod
    def from_folder(cls, base):
        return cls(
            join(base, 'train.csv'),
            join(base, 'train_images'),
            join(base, 'test_images'),
            join(base, 'car_models_json')
        )

    def _load_car(self, name):
        with open(join(self.car_json_folder, name + '.json')) as f:
            data = json.load(f)
        faces = np.array(data['faces']) - 1
        vertices = np.array(data['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        return CarData(
            car_type=data['car_type'],
            faces=faces,
            vertices=vertices
        )

    def _parse_labels_s(self, csv_file):
        with open(csv_file) as csv_file:
            data = csv.DictReader(csv_file)
            for row in data:
                yield row['ImageId'], self._parse_labels(row['PredictionString'])

    @staticmethod
    def _parse_labels(label_string):
        return [
            CarLabel(int(car_id), *data)
            for car_id, *data in zip_longest(*(
                    [iter(map(float, label_string.split()))] * num_params
            ))
        ]
