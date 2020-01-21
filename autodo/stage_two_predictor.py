import numpy as np
import torch

from autodo.custom_models import ResNet, BasicBlock
from autodo.stage_two_dataset import StageTwoDataset


class StageTwoPredictor:
    def __init__(self, model_file, gpu=False):
        self.device = torch.device('cuda:0' if gpu else 'cpu')
        self.net = ResNet(BasicBlock, [3, 3, 4, 3]).to(self.device)  # [2,2,2,2]
        self.net.load_state_dict(torch.load(model_file))
        self.net.eval()

    def predict(self, image_files, boxes):
        image_data = np.array([StageTwoDataset.load_image(image, box) for image, box in zip(image_files, boxes)])
        images = torch.from_numpy(image_data).float().to(self.device)
        outputs = self.net(images)
        return [StageTwoDataset.decode_label(output) for output in outputs]
