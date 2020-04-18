import imageio
import matplotlib
import torch
import argparse

#from networkx.drawing.tests.test_pylab import plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torchvision.utils import save_image

from unet import Unet
from dataset import LiverDataset
import os

from torchvision.utils import save_image
# 是否使用cuda
device = torch.device( "cpu")
#"cuda" if torch.cuda.is_available() else
x_transforms = transforms.Compose([#torchvision.transforms是pytorch中的图像预处理包
    # 一般用Compose把多个步骤整合到一起：
    transforms.ToTensor(),#convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
# 标准化至[-1,1],规定均值和标准差
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#Normalized an tensor image with mean and standard deviation
])
# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=2):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0#minibatch数
        for x, y in dataload:# 分100次遍历数据集，每次遍历batch_size=4
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()#每次minibatch都要将梯度(dw,db,...)清零
            # 把梯度置零，也就是把loss关于weight的导数变成0
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#梯度下降,计算出梯度
            optimizer.step()#更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)#保存模型
    return model

#训练模型
def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # shuffle = True,  # 乱序
    # num_workers = 2  # 多进程
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_model(model, criterion, optimizer, dataloaders)


#显示模型的输出结果
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))#加载模型
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)#batch_size默认为1
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        n = 0
        for x, _ in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()#对数据的维度进行压缩或者解压。Tensor转化为PIL图片
            from PIL import Image
            # image_array是归一化的二维浮点数矩阵
            img_y *= 255  # 变换为0-255的灰度值
            im = Image.fromarray(img_y)
            im = im.convert('1')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            matplotlib.image.imsave('%03d_predict.png'%n, im)
            #plt.imshow(img_y)
            n=n+1
        #     plt.pause(5)
        # plt.show()
        print("hello")


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
