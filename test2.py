'''
针对本地图片进行识别
'''
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from network import Network
from PIL import Image
import torchvision.transforms as transforms

# 定义图片的预处理转换
transform = transforms.Compose([
    transforms.Resize((28,28)),   # 将图片大小调整为模型输入的大小
    transforms.Grayscale(),   # 将图片转换为灰度图像
    transforms.ToTensor(),   # 将图片转换为张量
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./log/model.pt')
model.eval()

# 读取图片
image = Image.open('./test_dataset/0.jpg')
# 对图片进行预处理
input_image = transform(image)
input_image = input_image.unsqueeze(0) 
input_image = input_image.to(device)

with torch.no_grad():
    output = model(input_image)
    pred = output.data.max(1, keepdim=True)[1]
    predicted_label = pred.item()  # 获取预测结果的标签值
    print("Predicted label:", predicted_label)