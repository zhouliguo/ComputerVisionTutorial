from data_loader import LoadImagesAndLabels
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from network import VGG
from torch.optim.lr_scheduler import StepLR

def train(path, epochs, initial_lr, batch_size, workers, device):
    # 自定义dataset类，用于读取训练数据集
    train_dataset = LoadImagesAndLabels(path[0], training=True)
    # 实例化读取训练数据的迭代器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # 自定义dataset类，用于读取测试数据集
    val_dataset = LoadImagesAndLabels(path[1], training=False)
    # 实例化读取测试数据的迭代器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # 实例化CNN模型
    model = VGG(w=32, h=32).to(device)

    # 实例化损失函数（此处为交叉熵损失函数）
    criterion = nn.CrossEntropyLoss()
    # 实例化网络优化器（此处为随机梯度下降算法）
    optimizer = torch.optim.SGD(model.parameters(), initial_lr, momentum=0.9, weight_decay=1e-4)  # 优化方法，随机梯度下降
    # 实例化学习率更新策略，每调用scheduler.step() step_size次，将更新一次学习率
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # 开始训练循环
    for i in range(epochs):
        # 将CNN模型设置为train模式
        model.train()

        loss_sum = 0
        # 从训练数据集迭代器中循环读取数据，每次迭代会读取batch_size个样本
        for j, (images, labels) in enumerate(train_loader):
            # 将读取的数据送到训练设备上
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 将图片数据输入CNN模型，并得到预测的结果
            preds = model(images)

            # 计算预测结果与真实labels之间的差异(loss)
            loss = criterion(preds, labels)
            # loss反向传播
            loss.backward()

            optimizer.step()

            # 统计和打印loss信息
            loss_sum = loss_sum+loss.item()
            if (j+1)%100 == 0:
                print('Epoch ', i+1, '\tBatch ', j+1, '\tLoss ', loss_sum/100)
                loss_sum = 0

        # 学习率更新
        scheduler.step()

        # 将CNN模型设置为eval模式
        model.eval()

        # 用于记录总样本数
        num_total = 0
        # 用于记录被正确识别的样本数
        num_correct = 0

        # 从测试数据集迭代器中循环读取数据
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
    # 数据集路径：训练集和测试集
    data_path = ['../cifar_recognition/data/train/', '../cifar_recognition/data/test/']
    # 训练轮数
    epochs = 90
    # 初始学习率
    initial_lr = 0.01
    # 每次输入网络训练的样本数
    batch_size = 128
    # 读取数据的线程数
    workers = 8
    # 训练使用的设备
    device = torch.device('cuda:0')

    # 调用train函数，开始训练
    train(data_path, epochs, initial_lr, batch_size, workers, device)