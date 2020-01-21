import numpy as np
import torch
from os.path import isfile
from prettyparse import Usage
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from autodo.custom_models import ResNet, BasicBlock
from autodo.scripts.base_script import BaseScript
from autodo.stage_two_dataset import StageTwoDataset


class TrainStageTwoScript(BaseScript):
    usage = Usage('''
        Train the stage two network
        
        :model_file str
            Model file to load from/save to

        :-b --best-model-file
            Save best model file
    ''', best_model_file=lambda x: x.best_model_file or x.model_file) | StageTwoDataset.usage

    def run(self):
        args = self.args
        dataset = StageTwoDataset.from_args(args)
        train_stage_two(dataset, args.best_model_file, args.model_file)


main = TrainStageTwoScript.run_main


def train_stage_two(dataset, best_model_file, model_file):
    bestaccuracy = 0.9
    device = 'cudo:0' if torch.cuda.is_available() else 'cpu'
    net = ResNet(BasicBlock, [3, 3, 4, 3]).to(device)  # [2,2,2,2]
    net.train()
    for parameter in net.parameters():
        if len(parameter.shape) > 1:
            torch.nn.init.xavier_uniform_(parameter)
    if isfile(best_model_file):
        net.load_state_dict(torch.load(best_model_file))
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = AdamW(net.parameters(), lr=0.0001)
    scheduler = CyclicLR(
        optimizer, 0.000001, 0.0001, step_size_up=200,
        mode='triangular2', cycle_momentum=False, last_epoch=-1
    )
    L1 = torch.nn.L1Loss()
    BCE = torch.nn.BCEWithLogitsLoss()

    for epoch in range(50):
        running_accuracy = []
        for (images, targets) in tqdm(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            clsloss = BCE(outputs[:, 0], targets[:, 0])
            regloss = L1(outputs[:, 1:], targets[:, 1:])
            loss = clsloss + regloss
            cls_preds = np.greater(outputs[:, 0].cpu().detach().numpy(), 0)
            cls_truth = targets[:, 0].cpu().detach().numpy()
            correctness = np.equal(cls_preds, cls_truth).astype(int)
            accuracy = sum(correctness) / 64
            running_accuracy.append(accuracy)
            running_accuracy = running_accuracy[-10:]
            print(' clsloss ' + str(clsloss.cpu().detach().numpy())[:4] + ' regloss ' + str(
                regloss.cpu().detach().numpy())[:4] +
                  ' accuracy ' + str(np.mean(running_accuracy)), end='\r')
            if np.mean(running_accuracy) > bestaccuracy:
                bestaccuracy = np.mean(running_accuracy)
                torch.save(net.state_dict(), best_model_file)
                # print('totalloss', str(loss.detach().numpy())[:4], 'saved!', end = '\n')
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
