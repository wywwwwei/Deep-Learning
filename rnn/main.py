import argparse
from typing import Tuple
import torch
from torch.functional import split
from torch.utils.data import Dataset, DataLoader
from model.preprocess import Preprocessor
import model.lstm as lstm


def parse_arg() -> Tuple[str, str, str, str]:
    parser = argparse.ArgumentParser(description="ML2020 Spring HW4")

    parser.add_argument("-l", "--label", metavar="labeld data", default="./data/training_label.txt", help="the path of the labeld train data")
    parser.add_argument("-n", "--nolabel", metavar="unlabeld data", default="./data/training_nolabel.txt", help="the path of the unlabeld train data")
    parser.add_argument("-i", "--input", metavar="input file", default="./data/testing_data.txt", help="the path of the test data")
    parser.add_argument("-o", "--output", metavar="output file", default="./data/submit.csv", help="the path of the output file")

    args = parser.parse_args()
    return (args.label, args.nolabel, args.input, args.output)


class TextDataset(Dataset):

    def __init__(self, x, y) -> None:
        super().__init__()
        self.data = x
        self.label = y

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]


def main() -> None:
    label, nolabel, input, output = parse_arg()

    # read
    label_data = [line.strip('\n').split(' ') for line in open(label, encoding="UTF-8").readlines()]
    x_label = [line[2:] for line in label_data]
    y_label = [int(line[0]) for line in label_data]

    nolabel_data = open(nolabel, encoding="UTF-8").readlines()
    x_nolabel = [line.strip('\n').split(' ') for line in nolabel_data]

    test_data = ["".join(line.strip('\n').split(',')[1:]).strip() for line in open(input, encoding="UTF-8").readlines()[1:]]
    x_test = [line.split() for line in test_data]

    # [["word","word"],["word","word"]] -> LongTensor([[idx,idx],[idx,idx]])
    p = Preprocessor(x_label + x_nolabel)
    x_label = torch.stack([p.sentence_to_vector(x) for x in x_label])
    x_nolabel = torch.stack([p.sentence_to_vector(x) for x in x_nolabel])
    x_test = torch.stack([p.sentence_to_vector(x) for x in x_test])
    y_label = p.labels_to_tensor(y_label)

    train_size = int(0.9 * len(x_label))
    x_train, x_val = torch.split(x_label, [train_size, len(x_label) - train_size])
    y_train, y_val = torch.split(y_label, [train_size, len(y_label) - train_size])
    train_set = TextDataset(x_train, y_train)
    val_set = TextDataset(x_val, y_val)
    test_set = TextDataset(x_test, None)

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = lstm.LSTMClassifier(embedding=p.get_embedding(), hidden_size=150, num_layers=1)
    lstm.train(model, train_loader=train_loader, val_loader=val_loader, epochs=5, lr=0.001, threshold=0.5)

    # Semi-supervised Learning
    # Self training
    nolabel_set = TextDataset(x_nolabel, None)
    nolabel_loader = DataLoader(nolabel_set, batch_size=batch_size, shuffle=False, num_workers=4)
    y_nolabel = lstm.predict(model, nolabel_loader)

    self_train_size = int(0.9 * len(x_nolabel))
    x_self_train, x_self_val = torch.split(x_nolabel, [self_train_size, len(x_nolabel) - self_train_size])
    y_self_train, y_self_val = torch.split(y_nolabel, [self_train_size, len(y_nolabel) - self_train_size])

    self_train_set = TextDataset(x_self_train, y_self_train)
    self_val_set = TextDataset(x_self_val, y_self_val)

    self_train_loader = DataLoader(self_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    self_val_loader = DataLoader(self_val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    lstm.train(model, train_loader=self_train_loader, val_loader=val_loader, epochs=5, lr=0.001, threshold=0.5)
    # lstm.train(model, train_loader=self_train_loader, val_loader=self_val_loader, epochs=5, lr=0.001, threshold=0.5)

    # predict
    prediction = lstm.predict(model, test_loader, threshold=0.5).tolist()

    # output
    with open(output, mode='w', newline='') as sf:
        sf.write('id,label\n')
        for i, y in enumerate(prediction):
            sf.write('{},{}\n'.format(i, y))


if __name__ == "__main__":
    main()