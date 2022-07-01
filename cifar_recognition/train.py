from data_loader import LoadImagesAndLabels
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import DataLoader
from network import VGG
from torch.optim.lr_scheduler import StepLR

def train(path, epochs, initial_lr, batch_size, workers):
    train_dataset = LoadImagesAndLabels(path[0], training=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_dataset = LoadImagesAndLabels(path[0], training=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    model = VGG(w=32, h=32)

    criterion = nn.CrossEntropyLoss()   #loss function
    optimizer = torch.optim.SGD(model.parameters(), initial_lr, momentum=0.9, weight_decay=1e-4)  #优化方法，随机梯度下降
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for i in range(epochs):
        model.train()
        loss_sum = 0
        for j, (images, labels) in enumerate(train_loader):

            preds = model(images)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum+loss.item()
            if (j+1)%10 == 0:
                print('Epoch ', i+1, '\tBatch ', j+1, '\tLoss ', loss_sum/100)
                loss_sum = 0

        scheduler.step()

        model.eval()

        num_total = 0
        num_correct = 0
        for j, (images, labels) in enumerate(val_loader):
            num_total = num_total+len(images)

            preds = model(images)
            preds = F.softmax(preds, dim = 1)
            preds = torch.argmax(preds, dim=1)

            num_correct = num_correct + len(preds[preds==labels])
            

            



if __name__ == '__main__':
    train_path = 'cifar_recognition/data/train/'
    test_path = 'cifar_recognition/data/test/'

    epochs = 90
    initial_lr = 0.1
    batch_size = 8
    workers = 0

    train([train_path, test_path], epochs, initial_lr, batch_size, workers)