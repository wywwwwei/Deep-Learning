import argparse
import cv2
import numpy as np
import os
import torch
from torch.functional import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import model.cnn as cnn


def parse_arg() -> Tuple[str, str, str, str]:
    parser = argparse.ArgumentParser(description="ML2020 Spring HW3")

    parser.add_argument("-t", "--train", metavar="train data", default="./data/training", help="the directory of the train data")
    parser.add_argument("-v", "--validation", metavar="validation data", default="./data/validation", help="the directory of the validation data")
    parser.add_argument("-i", "--input", metavar="test data", default="./data/testing", help="the directory of the test data")
    parser.add_argument("-o", "--output", metavar="output_file", default="./data/testing/submit.csv", help="the path of the output file")

    args = parser.parse_args()
    return (args.train, args.validation, args.input, args.output)


def read_image(dir: str, label: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    images = sorted(os.listdir(dir))

    x = np.zeros((len(images), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(images)), dtype=np.uint8)

    for i, image in enumerate(images):
        data = cv2.imread(os.path.join(dir, image))
        x[i] = cv2.resize(data, (128, 128))  # resize to make all images are represented as (128,128,3)
        if label:
            y[i] = int(image.split('_')[0])

    return x, y


class ImgDataset(Dataset):

    def __init__(self, x: Tensor, y: Tensor = None, transform: transforms.Compose = None) -> None:
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def main() -> None:
    train_dir, validate_dir, input_dir, output_file = parse_arg()

    x_train, y_train = read_image(train_dir, label=True)
    x_validate, y_validate = read_image(validate_dir, label=True)
    x_test, _ = read_image(input_dir, label=False)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_validate shape: {x_validate.shape}")
    print(f"x_test shape: {x_test.shape}")

    # data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),            # (128,128,3) ndarray -> PIL.image
        transforms.RandomHorizontalFlip(),  # Horizontally flip the given image randomly with probability = 0.5
        transforms.RandomRotation(15),      # Rotate the image by angle = (-15,15)
        transforms.ToTensor(),              # PIL.image range in [0,255] -> FloatTensor (3,128,128) range in [0.0,1.0] 
    ])

    # test data doesn't need data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_set = ImgDataset(x_train, torch.LongTensor(y_train), train_transform)
    validate_set = ImgDataset(x_validate, torch.LongTensor(y_validate), test_transform)
    test_set = ImgDataset(x_test, transform=test_transform)

    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = cnn.CNNClassifier(output_num = 11)
    cnn.train(model, train_loader, validate_loader)
    prediction = cnn.predict(model, test_loader)

    # output
    with open(output_file, mode='w', newline='') as sf:
        sf.write('Id,Category\n')
        for i, y in enumerate(prediction):
            sf.write('{},{}\n'.format(i, y))


if __name__ == "__main__":
    main()