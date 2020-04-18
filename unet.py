import torch
from torch import nn#一般都是nn.Module的子类，可以借助nn.Module的父方法方便的管理各种需要的变量（状态）
#torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
#nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法
#需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。
#Linear的创建需要两个参数，inputSize 和 outputSize，inputSize：输入节点数，outputSize：输出节点数
class DoubleConv(nn.Module):#在Unet中绝大部分的conv都是两个conv连用的形式存在的，为了方便，我们可以先自定义一个double_conv类。
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()#调用父类的构造函数
        # 等价与nn.Module.__init__() nn.Module的子类函数必须在构造函数中执行父类的构造函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),#in_ch、out_ch是通道数
        # nn.Conv2d(in_channels，out_channels，kernel_size，stride = 1，padding = 0，dilation = 1，groups = 1，bias = True)
        # in_channels：输入维度
        # out_channels：输出维度
        # kernel_size：卷积核大小
        # stride：步长大小
        # padding：补0
        # padding的用途: 保持边界信息;可以对有差异的图片进行补齐,使得图像的输入大小一致;
        # 在卷积层中加入padding ,会使卷基层的输入维度与输出维度一致; 同时,可以保持边界信息
        # 其中padding补0 的策略是四周都补,如果padding=1,那么就会在原来输入层的基础上,上下左右各补一行,
         # 如果padding=(1,1)中第一个参数表示在高度上面的padding,第二个参数表示在宽度上面的padding
        # dilation：kernel间距
            nn.BatchNorm2d(out_ch),#BatchNorm2d最常用于卷积网络中(防止梯度消失或爆炸)，设置的参数就是卷积的输出通道数
            #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
            # 这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
#由于weights是可以训练的，所以使用Parameter来定义
    # 和tensorflow不一样，pytorch中模型的输入是一个Variable，而且是Variable在图中流动，不是Tensor。
    # 这可以从forward中每一步的执行结果可以看出
    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)   #每次把图像尺寸缩小一半
#卷积操作中 pool层是比较重要的，是提取重要信息的操作，可以去掉不重要的信息，减少计算开销

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)#kerner_size(int or tuple) - 卷积核的大小
        # stride卷积步长，即要将输入扩大的倍数

        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)

        c2=self.conv2(p1)
        p2=self.pool2(c2)

        c3=self.conv3(p2)
        p3=self.pool3(c3)

        c4=self.conv4(p3)
        p4=self.pool4(c4)

        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)#按维数1（列）拼接,列增加
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        #out = nn.Sigmoid()(c10)#化成(0~1)区间
        return c10









