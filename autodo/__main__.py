from collections import namedtuple

import cv2
import gluoncv as gcv
import numpy as np
import pandas as pd
import re
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from autodo.dataset import Dataset


def draw_obj(image, vertices, triangles):
    for t in triangles:
        coord = np.array([vertices[t[0]], vertices[t[1]], vertices[t[2]]], dtype=np.int32)
        cv2.polylines(image, np.int32([coord]), 1, (0, 0, 255))


def extract_image_id(filename):
    m = re.match(r'.*(ID_[0-9a-f]{9}).jpg', filename)
    return m.group(1) if m else None


def plot_data(img, bboxes, scores, box_ids, image_id=None, dataset=None):
    ax = gcv.utils.viz.plot_bbox(img, bboxes[0], scores[0], box_ids[0], class_names=['car'])
    overlay = np.zeros_like(img)
    if image_id and dataset:
        image_id = m.group(1)
        for label in dataset.labels_s[image_id]:
            pix_pos = dataset.calc_verts(label)
            height = img.shape[0]
            width = img.shape[1]
            pix_pos[:, 0] *= width
            pix_pos[:, 1] *= height
            draw_obj(overlay, pix_pos, dataset.car_models[label.car_id].faces)
            bbox = dataset.calc_bbox(label)
            bbox[[0, 2]] *= width
            bbox[[1, 3]] *= height
            xmin, ymin, xmax, ymax = b = [int(x) for x in bbox]
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor='#00ff00',
                                 linewidth=1.5)
            ax.add_patch(rect)
    alpha = .5
    img = np.array(img)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    plt.imshow(img)
    plt.show()


def main():
    parser = ArgumentParser(description='Car position predictor tool')
    parser.add_argument('data_folder', help='Dataset folder with labels')
    parser.add_argument('out_csv', help='Output csv')
    parser.add_argument('-g', '--gpu', help='Enable GPU', action='store_true')
    args = parser.parse_args()
    dataset = Dataset.from_folder(args.data_folder)
    import mxnet as mx
    ctx = mx.gpu(0) if args.gpu else mx.cpu()
    net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx=ctx)
    car_id = net.classes.index('car')

    DataRow = namedtuple('DataRow', 'image_id score xmin ymin xmax ymax')
    rows = []

    try:
        for image in dataset.get_train_file_names():
            image_id = extract_image_id(image)
            x, img = gcv.data.transforms.presets.rcnn.load_test(image)
            box_ids, scores, bboxes = net(x.as_in_context(ctx))
            car_indices = (np.squeeze(box_ids[0].asnumpy()) == car_id)
            scores = np.squeeze(scores[0].asnumpy()[car_indices])
            bboxes = bboxes[0].asnumpy()[car_indices]
            print('Found {} cars.'.format(len(bboxes)))

            for bbox, score in zip(bboxes, scores):
                rows.append(DataRow(image_id, score, *bbox))
    except KeyboardInterrupt:
        print()
    finally:
        data = pd.DataFrame(data=rows)
        data.to_csv(args.out_csv)


if __name__ == '__main__':
    main()
