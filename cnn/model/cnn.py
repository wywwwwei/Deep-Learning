import torch
import torch.nn as nn
from torch.optim import Adam
import time
from torch.utils.data.dataloader import DataLoader
from typing import List


class CNNClassifier(nn.Module):

    def __init__(self, output_num: int):
        super().__init__()

        # input [3,128,128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # [64,128+2-3+1,128+2-3+1] = [64,128,128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [64,128/2,128/2] = [64,64,64]
            nn.Conv2d(64, 128, 3, 1, 1),    # [128,64+2-3+1,64+2-3+1] = [128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [128,64/2,64/2] = [128,32,32]
            nn.Conv2d(128, 256, 3, 1, 1),   # [256,32+2-3+1,32+2-3+1] = [256,32,32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [256,32/2,32/2] = [256,16,16]
            nn.Conv2d(256, 512, 3, 1, 1),   # [512,16+2-3+1,16+2-3+1] = [512,16,16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512,16/2,16/2] = [512,8,8]
            nn.Conv2d(512, 512, 3, 1, 1),   # [512,8+2-3+1,8+2-3+1] = [512,8,8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512,8/2,8/2] = [512,4,4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_num),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 512 * 4 * 4)
        return self.fc(out)


def train(model: CNNClassifier, train_loader: DataLoader, validate_loader: DataLoader, epochs: int = 30, lr: float = 0.001) -> None:
    model = model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for data in train_loader:
            images, labels = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            output = model(images)
            batch_loss = loss(output, labels)
            batch_loss.backward()
            optimizer.step()

            _, predict = torch.max(output.data, dim=1)
            train_acc += torch.sum(predict == labels).item()
            train_loss += batch_loss.item()

        model.eval()
        for data in validate_loader:
            images, labels = data[0].cuda(), data[1].cuda()
            output = model(images)
            batch_loss = loss(output, labels)

            _, predict = torch.max(output.data, dim=1)
            val_acc += torch.sum(predict == labels).item()
            val_loss += batch_loss.item()
            
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, epochs, time.time()-epoch_start_time, \
             train_acc/train_loader.dataset.__len__(), train_loss/train_loader.dataset.__len__(), \
             val_acc/validate_loader.dataset.__len__(), val_loss/validate_loader.dataset.__len__()))


def predict(model: CNNClassifier, x: DataLoader) -> List[int]:
    model = model.cuda()
    prediction = []
    model.eval()
    with torch.no_grad():
        for images in x:
            output = model(images.cuda())
            _, predict = torch.max(output.data, dim=1)
            prediction.extend(predict.tolist())
    return prediction