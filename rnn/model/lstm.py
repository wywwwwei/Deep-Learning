import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader


class LSTMClassifier(nn.Module):

    def __init__(self, embedding: torch.FloatTensor, hidden_size: int, num_layers: int, dropout: float = 0.5, fix_embedding: bool = True) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True

        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.LongTensor):
        x_vec = self.embedding(x)
        out, _ = self.lstm(x_vec)
        last_state = out[:, -1, :]
        return self.classifier(last_state)


def train(model: LSTMClassifier, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, threshold: float = 0.5) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"start training, parameter total:{total}, trainable:{trainable}")

    model = model.cuda()
    loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    train_size = train_loader.dataset.__len__()
    val_size = val_loader.dataset.__len__()

    for epoch in range(epochs):
        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0
        model.train()
        for data in train_loader:
            text, label = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()

            output = model(text).squeeze()
            predict = (output >= threshold).long()

            cur_loss = loss(output, label.float())
            cur_loss.backward()
            optimizer.step()

            train_acc += torch.sum(predict == label).item()
            train_loss += cur_loss.item()

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                text, label = data[0].cuda(), data[1].cuda()

                output = model(text).squeeze()
                predict = (output >= threshold).long()

                cur_loss = loss(output, label.float())

                val_acc += torch.sum(predict == label).item()
                val_loss += cur_loss.item()

        print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, epochs, train_acc/train_size, train_loss/train_size, val_acc/val_size, val_loss/val_size))


def predict(model: LSTMClassifier, x: DataLoader, threshold: float = 0.5) -> torch.LongTensor:
    result = []
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for data in x:
            text = data.cuda()

            output = model(text).squeeze()
            predict = (output >= threshold).long()
            result.append(predict)
    return torch.cat(result, 0)
