# ResNet猫狗图片分类

[源码下载地址](https://github.com/1KUnn/ResNet)

python 3.7.4

CUDA 12.6

## 在Linux服务器创建环境

```python
#创建环境变量
conda create -n medical python=3.7.4 -y

#激活环境
conda activate medical

#检查PyTorch 是否正确安装
python -c "import torch; print(torch.__version__)"
```

正确安装的结果如下

![image-20250318192206369](F:\Typora图\image-20250318192206369.png)



## 下载数据集

新建py文件 2.py 内容如下 

```python
import kagglehub
# 下载kaggle的数据集
# Download latest version
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

# 打印下载文件的位置
print("Path to dataset files:", path)
```



在medical环境里运行 

```python
python 2.py
```

![image-20250318192729794](F:\Typora图\image-20250318192729794.png)

箭头处就是下载的数据集文件地址



将数据集转移到你想要存放的目录下 我这是/home/huangrw/autodl-tmp/pytorchProject/data/ResNet

```python
# 创建目标目录 地址可以改自己的
mkdir -p /home/huangrw/autodl-tmp/pytorchProject/data/ResNet

# 复制所有文件和子目录
cp -r /home/huangrw/.cache/kagglehub/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/versions/1/PetImages/* /home/huangrw/autodl-tmp/pytorchProject/data/ResNet/
```

![image-20250318192935937](F:\Typora图\image-20250318192935937.png)

成功



## 模型部分

**处理过程：*原始图片 -> ResNet处理 -> 1000个原始分数 -> softmax转换 -> 概率分布 -> 选择最高概率***

### 确定图片预处理数据

```python
resnet = models.resnet101(pretrained=True) 
#下载预训练后的101层的resnet

# 打印模型架构
print(resnet)

# 查看第一个卷积层的详细参数
first_conv = resnet.conv1
print("第一个卷积层参数:")
print(f"输入通道数: {first_conv.in_channels}")  # 应该显示 3
print(f"输出通道数: {first_conv.out_channels}")
print(f"卷积核大小: {first_conv.kernel_size}")
print(f"步长: {first_conv.stride}")
print(f"padding: {first_conv.padding}")
```

![image-20250320064334559](F:\Typora图\image-20250320064334559.png)
$$
输出尺寸 = [(W - K + 2P) / S] + 1\\
其中：
W = 输入尺寸\
K = 卷积核大小\
P = padding大小\
S = 步长\\
代入有[(224 - 7 + 2*3) / 2] + 1 = 112\\
后续计算有112->56->28->14->7 特征图大小始终保持整数
$$


### 数据预处理

```python
#因为数据集里的图片不是统一格式的，所以需要对图片进行预处理（统一格式）
perprocess = transforms.Compose([
    transforms.Resize(256), #将图片缩放到 256x256 像素 选择 256 是一个经验值，它足够大以保留重要特征，同时又不会占用太多内存
    transforms.CenterCrop(224), #从256x256的图片中心裁剪出224x224的区域(224x224 是 ResNet 模型的标准输入大小)
    transforms.ToTensor(), #将图片转换为tensor张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],#这些值是在 ImageNet 数据集上计算出来的均值和标准差 对应 RGB 三个通道
                         std=[0.229, 0.224, 0.225]) #ResNet 模型是在 ImageNet 上预训练的 用相同的归一化参数可以获得最佳效果
])

```

**为什么不直接Resize为224**：resize 到 256 再裁剪能保持图像原有比例、模仿人类观察物体的方式（关注中心区域）和是 ImageNet 训练中使用的标准预处理方法。



### 第一张图片

```python
# 加载一张图片
img = Image.open("/home/huangrw/autodl-tmp/pytorchProject/data/ResNet/Dog/0.jpg")
# 对图片进行预处理
img = perprocess(img)
# 增加一个维度，因为模型的输入是一个四维张量 批量输入提升效率
batch_img = torch.unsqueeze(img, dim=0) #dim=0 表示在第 0 维增加一个维度 [3, 224, 224]->[1, 3, 224, 224]
resnet.eval() #将模型设置为评估模式
# 将图片输入模型
out = resnet(batch_img)
# 打印输出结果
print(out)
```



![image-20250320073029668](F:\Typora图\image-20250320073029668.png)

返回的值其实就是resnet对这个像素点分类预测值



### 测试

为模型添加标签信息 [下载地址](https://github.com/1KUnn/ResNet)

![image-20250320082514796](F:\Typora图\image-20250320082514796.png)

```python
# 读取标签文件 并加载映射关系
with open("/home/huangrw/autodl-tmp/pytorchProject/data/ResNet/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
# 获取预测结果
_, index = torch.max(out, 1) #获取最大值的索引 _是最大值，index是索引
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 #将输出结果转换为概率 dim=1取各个样本自己的概率分布
# 打印预测结果
print(labels[index[0]], percentage[index[0]].item()) 

```

运行后发现模型有54.55%的把握认为输入图片是斑点狗

![image-20250320082839863](F:\Typora图\image-20250320082839863.png)

