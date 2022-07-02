from data_loader import LoadImagesAndLabels
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from network import VGG
from torch.optim.lr_scheduler import StepLR

def train(path, epochs, initial_lr, batch_size, workers, device):
    train_dataset = LoadImagesAndLabels(path[0], training=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_dataset = LoadImagesAndLabels(path[1], training=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    model = VGG(w=32, h=32).to(device)

    criterion = nn.CrossEntropyLoss()   #loss function
    optimizer = torch.optim.SGD(model.parameters(), initial_lr, momentum=0.9, weight_decay=1e-4)  # 优化方法，随机梯度下降
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for i in range(epochs):
        model.train()
        loss_sum = 0
        for j, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(images)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum+loss.item()
            if (j+1)%100 == 0:
                print('Epoch ', i+1, '\tBatch ', j+1, '\tLoss ', loss_sum/100)
                loss_sum = 0

        scheduler.step()

        model.eval()

        num_total = 0
        num_correct = 0
        for j, (images, labels) in enumerate(val_loader):
            num_total = num_total+len(images)

            images = images.to(device)
            preds = model(images)

            preds = F.softmax(preds, dim = 1)
            preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
            labels = labels.numpy()

            num_correct = num_correct + len(preds[preds==labels])
        print('Accuracy: ', num_correct/num_total)

if __name__ == '__main__':
    data_path = ['/opt/data/liguo/ComputerVisionTutorial/cifar_recognition/data/train/', '/opt/data/liguo/ComputerVisionTutorial/cifar_recognition/data/test/']
    epochs = 90
    initial_lr = 0.01
    batch_size = 128
    workers = 8
    device = torch.device('cuda:0')

    train(data_path, epochs, initial_lr, batch_size, workers, device)