import numpy as np
import torch
from os.path import isfile
from prettyparse import Usage
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from autodo.scripts.base_script import BaseScript
from autodo.stage_three_dataset import StageThreeDataset
from autodo.stage_three_model import MyUNet


class TrainStageThreeScript(BaseScript):
    usage = Usage('''
        Train the stage three network
        
        :model_file str
            Model file to load from/save to

        :-b --best-model-file
            Save best model file
    ''', best_model_file=lambda x: x.best_model_file or x.model_file) | StageThreeDataset.usage

    def run(self):
        args = self.args
        dataset = StageThreeDataset.from_args(args)
        train_stage_three(dataset, args.best_model_file, args.model_file)


main = TrainStageThreeScript.run_main


def train_stage_three(dataset, best_model_file, model_file):
    bestaccuracy = 0.9
    device = 'cudo:0' if torch.cuda.is_available() else 'cpu'
    net = MyUNet(3, device).to(device)
    net.train()
    for parameter in net.parameters():
        if len(parameter.shape) > 1:
            torch.nn.init.xavier_uniform_(parameter)
    if isfile(best_model_file):
        net.load_state_dict(torch.load(best_model_file))
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(net.parameters(), lr=0.0001)
    scheduler = CyclicLR(
        optimizer, 0.000001, 0.0001, step_size_up=200,
        mode='triangular2', cycle_momentum=False, last_epoch=-1
    )
    L1 = torch.nn.L1Loss(size_average=False)

    for epoch in range(50):
        for (images, targets, out_masks) in tqdm(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            out_masks = out_masks.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = L1(outputs * out_masks, targets * out_masks) / 4
            outputs = (outputs * out_masks).cpu().detach().numpy()
            targets = (targets * out_masks).cpu().detach().numpy()
            if np.mean(np.linalg.norm(targets * 100, axis=1)) > 0:
                truth_norm = np.linalg.norm(targets * 100, axis=1).flatten()
                error_norm = np.linalg.norm(outputs * 100 - targets * 100, axis=1).flatten()
                truth_norm, error_norm = truth_norm[truth_norm > 0], error_norm[error_norm > 0]
                accuracy = sum((error_norm / truth_norm) < 0.1) / len(error_norm)
                print('mean error', np.mean(error_norm / truth_norm), 'accuracy', accuracy, end='\t')
            else:
                accuracy = 0.0
            print('L1loss', loss.cpu().detach().numpy(), end='\r')
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                torch.save(net.state_dict(), best_model_file)
            else:
                pass
                # print('totalloss', str(loss.detach().numpy())[:4]+' ', end = '\n')
            loss.backward()
            optimizer.step()
            scheduler.step(None)
            # if idx%5==0:
            #    print('\n', outputs[0].cpu().detach().numpy(), targets[0].cpu().detach().numpy(), '\n')
            # idx+=1
        torch.save(net.state_dict(), model_file)
        print(epoch)
