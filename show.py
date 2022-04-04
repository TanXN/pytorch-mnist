import enum
import optparse
from os import access
from model import NeuralNetwork
import torch
from dataset import ImageDataset
from dataset import DataLoader
import matplotlib.pyplot as plt
import numpy as np

trian_image_set = 'data/train-images.idx3-ubyte'
train_label = 'data/train-labels.idx1-ubyte'

test_iamge_set = 'data/t10k-images.idx3-ubyte'
test_label = 'data/t10k-labels.idx1-ubyte'


if __name__ == "__main__":
    model = NeuralNetwork()
    model.load_state_dict(torch.load('model_weight.pth'))
    model.eval()

    train_data = ImageDataset(trian_image_set, train_label)
    test_data = ImageDataset(test_iamge_set, test_label)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_imgs, train_labels = next(iter(train_dataloader))
    print(f"Image batch shape: {train_imgs.size()}")
    print(f"Labels batch shape: {train_labels.size()}" )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    # To SEE it
    # for batch,(X,y) in enumerate(test_dataloader):
    #     X = X.float()
    #     y = y.long()
    #     pred = model(X)
    #     # print(pred)
    #     max = pred.argmax(1)
    #     print(np.array(max)[0])
    #     plt.imshow(X.reshape(X.shape[1],X.shape[2]))
    #     plt.show()

    # Compute accuracy
    total,correct = 0,0
    for batch,(X,y) in enumerate(test_dataloader):
        X = X.float()
        y = y.long()
        pred = model(X)
        total += X.size(0)
        correct += (pred.argmax(1)==y).float().sum()

    print(f"accuracy: {correct / total}")