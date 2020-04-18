from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    imgs=[]#列表
    n=len(os.listdir(root))//2#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    #处理的图片的组数，//整数除法
    for i in range(n):
        img=os.path.join(root,"%03d.png"%i)#连接两个或更多的路径名组件
        mask=os.path.join(root,"%03d_mask.png"%i)#赋值给某个变量
        imgs.append((img,mask))
    return imgs


class LiverDataset(Dataset):

    #创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):#root表示图片路径
        #一个函数，输入为target，输出对其的转换。例子，输入的是图片标注的string，输出为word的索引。
        #transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
        #target_transform：对label的转换
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):#可以让对象实现迭代功能#如果类把某个属性定义为序列，可以使用__getitem__()输出序列属性中的某个元素
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y#返回的是图片

    def __len__(self):
        return len(self.imgs)#400,list[i]有两个元素，[img,mask]
class LiverDataset1(Dataset):

    #创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None):#root表示图片路径
        #一个函数，输入为target，输出对其的转换。例子，输入的是图片标注的string，输出为word的索引。
        #transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
        #target_transform：对label的转换
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform


    def __getitem__(self, index):#可以让对象实现迭代功能#如果类把某个属性定义为序列，可以使用__getitem__()输出序列属性中的某个元素
        x_path = self.imgs[index]
        img_x = Image.open(x_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x#返回的是图片

    def __len__(self):
        return len(self.imgs)#400,list[i]有两个元素，[img,mask]
