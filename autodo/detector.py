from collections.__init__ import namedtuple

import numpy as np
from abc import abstractmethod


class CarBoxDetector:
    @abstractmethod
    def predict_boxes(self, image_filename):
        pass


class GluonDetector(CarBoxDetector):
    def __init__(self, gpu=False):
        import mxnet as mx
        import gluoncv as gcv
        self.ctx = mx.gpu(0) if gpu else mx.cpu()
        self.net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx=self.ctx)
        self.car_id = self.net.classes.index('car')

    def predict_boxes(self, image_filename):
        import gluoncv as gcv
        x, img = gcv.data.transforms.presets.rcnn.load_test(image_filename)
        box_ids, scores, bboxes = self.net(x.as_in_context(self.ctx))
        car_indices = (np.squeeze(box_ids[0].asnumpy()) == self.car_id)
        scores = np.squeeze(scores[0].asnumpy()[car_indices])
        bboxes = bboxes[0].asnumpy()[car_indices]
        bboxes[[0, 2]] /= img.shape[0]
        bboxes[[1, 3]] /= img.shape[1]
        return [Box(score, *bbox) for score, bbox in zip(scores, bboxes)]


class PyTorchDetector(CarBoxDetector):
    def __init__(self, gpu=False):
        import torch
        import torchvision
        model_t = torchvision.models.detection.fasterrcnn_resnet50_fpn
        self.device = torch.device('cuda:0' if gpu else 'cpu')
        self.model = model_t(pretrained=True, box_nms_thresh=0.3).to(self.device)
        self.model.eval()

    # noinspection PyArgumentList
    def predict_boxes(self, image_filename):
        import torch
        import matplotlib.pylab as plt
        input_tensor = torch.zeros(1, 3, 2710, 3384).to(self.device)
        img = plt.imread(image_filename).astype('float32') / 255
        input_tensor[0] = torch.Tensor(img.transpose((2, 0, 1)))
        output = self.model(input_tensor)[0]
        bboxes = output['boxes']
        labels = output['labels']
        scores = output['scores']
        bboxes[[0, 2]] /= img.shape[0]
        bboxes[[1, 3]] /= img.shape[1]
        return [
            Box(float(confidence), *map(float, box))
            for box, label, confidence in zip(bboxes, labels, scores)
            if label == 3
        ]


Box = namedtuple('Box', 'confidence xmin ymin xmax ymax')