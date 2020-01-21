from collections import namedtuple

from argparse import ArgumentParser

from autodo.dataset import Dataset
from autodo.detector import PyTorchDetector
from autodo.utils import extract_image_id, iou

DataRow = namedtuple('DataRow', 'image_id box_id score xmin ymin xmax ymax')


def overlaps_own_car(box: tuple):
    own_car = [600 / 3384, 2500 / 2710, 3384 / 3384, 2710 / 2710]
    return iou(own_car, box) != 0


def main():
    parser = ArgumentParser(description='Predict car bounding boxes in images')
    parser.add_argument('data_folder', help='Dataset folder with labels')
    parser.add_argument('out_csv', help='Output csv')
    parser.add_argument('-g', '--gpu', help='Enable GPU', action='store_true')
    parser.add_argument('-t', '--test', help='Run on test data', action='store_true')
    parser.add_argument('-r', '--remove-extra', help='Remove extraneous boxes', action='store_true')
    args = parser.parse_args()

    rows = []
    dataset = Dataset.from_folder(args.data_folder)
    detector = PyTorchDetector(args.gpu)
    file_names = dataset.get_test_file_names() if args.test else dataset.get_train_file_names()

    try:
        for image in file_names:
            image_id = extract_image_id(image)
            boxes = [
                i
                for i in detector.predict_boxes(image)
                if not overlaps_own_car(i[1:]) and i.confidence >= 0.5
            ]
            ordered_boxes = []
            if args.test:
                ordered_boxes = boxes
            else:
                real_boxes = [dataset.calc_bbox(i) for i in dataset.labels_s[image_id]]
                for i in real_boxes:
                    if not boxes:
                        break
                    best_match = min(range(len(boxes)), key=lambda x: iou(i, boxes[x][1:]))
                    overlap = iou(i, boxes[best_match][1:])
                    if overlap > 0:
                        ordered_boxes.append(boxes.pop(best_match))
                if boxes and not args.remove_extra:
                    ordered_boxes.extend(boxes)

            print('Found {} cars.'.format(len(ordered_boxes)))

            for box_id, network_data in enumerate(ordered_boxes):
                rows.append(DataRow(image_id, box_id, *network_data))
    except KeyboardInterrupt:
        print()
    finally:
        import pandas as pd
        data = pd.DataFrame(data=rows)
        data.to_csv(args.out_csv)


if __name__ == '__main__':
    main()
