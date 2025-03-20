import torch
from torchvision import models, transforms
from PIL import Image

#下载预训练后的101层的resnet
resnet = models.resnet101(pretrained=True)

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

#因为数据集里的图片不是统一格式的，所以需要对图片进行预处理（统一格式）
perprocess = transforms.Compose([
    transforms.Resize(256), #将图片缩放到 256x256 像素 选择 256 是一个经验值，它足够大以保留重要特征，同时又不会占用太多内存
    transforms.CenterCrop(224), #从256x256的图片中心裁剪出224x224的区域(224x224 是 ResNet 模型的标准输入大小)
    transforms.ToTensor(), #将图片转换为tensor张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],#这些值是在 ImageNet 数据集上计算出来的均值和标准差 对应 RGB 三个通道
                         std=[0.229, 0.224, 0.225]) #ResNet 模型是在 ImageNet 上预训练的 用相同的归一化参数可以获得最佳效果
])


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


# 读取标签文件 并加载映射关系
with open("/home/huangrw/autodl-tmp/pytorchProject/data/ResNet/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
# 获取预测结果
_, index = torch.max(out, 1) #获取最大值的索引 _是最大值，index是索引
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 #将输出结果转换为概率 dim=1取各个样本自己的概率分布
# 打印预测结果
print(labels[index[0]], percentage[index[0]].item()) 

_, index = torch.sort(out, descending=True) #对输出结果进行排序
[(labels[i], percentage[i].item()) for i in index[0][:5]] #打印前五个预测结果
