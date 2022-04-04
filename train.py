
from tkinter.tix import Y_REGION
from dataset import ImageDataset
from model import NeuralNetwork
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch

trian_image_set = 'data/train-images.idx3-ubyte'
train_label = 'data/train-labels.idx1-ubyte'

test_iamge_set = 'data/t10k-images.idx3-ubyte'
test_label = 'data/t10k-labels.idx1-ubyte'

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.float()
        y = y.long()
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num__batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader: 
            X = X.float()
            y = y.long()
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num__batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    train_data = ImageDataset(trian_image_set, train_label)
    test_data = ImageDataset(test_iamge_set, test_label)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_imgs, train_labels = next(iter(train_dataloader))
    print(f"Label batch shape: {train_imgs.size()}")
    print(f"Labels batch shape: {train_labels.size()}" )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)
    # Begin Train 

    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------")
        train_loop(train_dataloader,model,loss_fn,optimizer)
        test_loop(test_dataloader,model,loss_fn)
    print("Done!")
    torch.save(model.state_dict(), 'model_weight.pth')

