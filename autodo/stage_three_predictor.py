import numpy as np
import torch

from autodo.stage_three_dataset import StageThreeDataset
from autodo.stage_three_model import MyUNet


class StageThreePredictor:
    def __init__(self, model_file, gpu=False):
        self.device = torch.device('cuda:0' if gpu else 'cpu')
        self.net = MyUNet(3, self.device).to(self.device)
        self.net.load_state_dict(torch.load(model_file))
        self.net.eval()

    def predict(self, filenames, xcenters_per_image, ycenters_per_image):
        input_data = np.array([
            StageThreeDataset.make_input(filename, xcenters, ycenters)
            for filename, xcenters, ycenters in zip(filenames, xcenters_per_image, ycenters_per_image)
        ])
        input_tensor = torch.from_numpy(input_data).float().to(self.device)
        outputs = self.net(input_tensor)
        return [
            StageThreeDataset.decode_output(output, xcenters, ycenters)
            for output, xcenters, ycenters in zip(outputs, xcenters_per_image, ycenters_per_image)
        ]
