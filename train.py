import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from network import Network
from torch import nn

trans = transforms.ToTensor()

train_dataset = torchvision.datasets.FashionMNIST('./dataset', train=True, transform=trans, download=False)
test_dataset = torchvision.datasets.FashionMNIST('./dataset', train=False, transform=trans, download=False)

train_loader = DataLoader(train_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Network().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

epoch = 10
total_train_step = 0

for i in range(epoch):
    print("===============第{}轮训练=================".format(i+1))
    for data in train_loader:
        image , targets = data
        image = image.to(device)
        targets = targets.to(device)

        output = net(image)
        loss = loss_fn(output,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            print("训练次数:{},Loss:{}".format(total_train_step,loss))

torch.save(net, './log/model.pt')



