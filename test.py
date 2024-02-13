'''
针对测试集图片进行识别验证
'''
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from network import Network
from PIL import Image
import numpy
import torchvision.transforms as transforms
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

trans = transforms.ToTensor()

test_dataset = torchvision.datasets.FashionMNIST('./dataset', train=False, transform=trans, download=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = DataLoader(test_dataset, batch_size=8)

prediction = []
labels = []
test_losses = []
correct = 0
test_loss = 0

model = torch.load('./log/model.pt')
model.eval()

with torch.no_grad():
    for image, target in test_loader:
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        # 累加损失，调用.item()可以从张量中提取元素的值
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        print(pred)
        
test_loss /= len(test_loader.dataset)
test_losses.append(test_loss)
# 打印结果
print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. *correct / len(test_loader.dataset)
    ))       
