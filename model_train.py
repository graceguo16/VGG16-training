import copy
import time

import pandas as pd
import torch
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import VGG16
import torch.nn as nn
import pandas

#数据加载函数
def train_val_data_process():
    train_data = FashionMNIST(root="./data",
                              train = True,
                              transform = transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                              download= True)

    train_data, val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset = train_data,
                                       batch_size = 32,
                                       shuffle = True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader,train_data,val_dataloader

#模型训练函数
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #使用Adam优化器，学习率0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    #损失函数 分类用交叉熵
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    #训练集损失列表
    train_loss_all = []
    #验证集损失列表
    val_loss_all = []
    #训练集准确度列表
    train_acc_all = []
    #验证集准确度列表
    val_acc_all = []

    since = time.time()


    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*10)


        #初始化参数
        train_loss = 0.0
        #训练集准确度
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        #样本数量（验证集+训练集）
        train_num = 0
        val_num = 0

        #对每个mini—batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            #特征放到训练设备中
            b_x = b_x.to(device)
            #标签放到训练设备中
            b_y = b_y.to(device)
            #设置模型为训练mode
            model.train()

            output = model(b_x)#forward传播

            #查找每一行最大值的标softmax
            pre_lab = torch.argmax(output,dim=1)
            #计算每一个batch的损失函数
            loss = criterion(output,b_y)
            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #根据梯度信息更新参数，降低loss函数计算值
            optimizer.step()
            #对损失函数进行累加
            train_loss += loss.item()* b_x.size(0)
            #如果预测正确，则准确度+1
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num+= b_x.size(0)


        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            # 标签放到验证设备中
            b_y = b_y.to(device)
            #评估模式
            model.eval()

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度+1
            val_corrects += torch.sum(pre_lab == b_y.data)
            #当前验证样本数量
            val_num += b_x.size(0)

        #计算并保存训练集的loss值
        train_loss_all.append(train_loss/train_num)
        #计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item()/train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))



        #寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())
        #计算耗时
        time_use = time.time()- since
        print("训练和验证耗费的时间{:.0f}m{:0f}s".format(time_use//60,time_use%60))

        #选择最优参数
        #加载最高准确率下的模型参数
    # 加载最优权重到模型
    model.load_state_dict(best_model_wts)
    # 保存 state_dict 到文件
    torch.save(best_model_wts, "/Users/guoyumeng/Desktop/VGG16/best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                           "train_loss_all": train_loss_all,
                                           "val_loss_all": val_loss_all,
                                           "train_acc_all": train_acc_all,
                                           "val_acc_all": val_acc_all})

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)#一行两列的第一张图
    plt.plot(train_process["epoch"],train_process.train_loss_all,"ro-",label = "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__ == "__main__":
    VGG16 = VGG16()
    train_dataloader, _,val_dataloader = train_val_data_process()
    train_process = train_model_process(VGG16, train_dataloader, val_dataloader, num_epochs=20)
    matplot_acc_loss(train_process)


    #模型实例化



#如果 batch_size=100，数据集大小=1000：epoch是整个数据集被完整训练一次的轮数
#1 epoch = 10 iterations
#5 epochs = 50 iterations





