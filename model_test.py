import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model_train import VGG16

def test_data_process():
    test_data = FashionMNIST(root="./data",
                              train = False,
                              transform = transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                              download= True)

    test_dataloader = Data.DataLoader(dataset = test_data,
                                       batch_size = 1,
                                       shuffle = True,
                                       num_workers= 0)

    return test_dataloader

#test_dataloader = test_data_process() 没问题！
def test_model_process(model,test_dataloader):
    #设置测试所用到的设备
    device = "cuda" if torch.cuda.is_available()else "cpu"
    #将模型放到训练设备中
    model = model.to(device)
    #初始化参数
    test_corrects = 0.0
    test_num = 0

    #只进行前向传播计算，不计算梯度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            #把特征和标签都放进设备
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            #forward,输出每个样本预测值
            output = model(test_data_x)
            #查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output,dim = 1)
            #如果越正确，则test corrects +1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            #将所有的测试样本累加
            test_num += test_data_x.size(0)
        #计算测试准确率
        # double:把 train_corrects 张量转换为 float64（double 类型）
        # item:把 PyTorch 张量（Tensor）转换成普通 Python 数值（float/int）
        test_acc = test_corrects.double().item()/test_num
        print('测试准确率为：',test_acc)

if __name__ == "__main__":
    model = VGG16()
    model.load_state_dict(torch.load('best_model.pth'))

        #加载测试数据
    test_dataloader = test_data_process()
        #加载模型测试的函数
    test_model_process(model,test_dataloader)

    device = "cuda" if torch.cuda.is_available()else "cpu"
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()

            print("预测值:", classes[result], "------", "真实值:", classes[label])














