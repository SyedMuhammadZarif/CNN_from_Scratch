import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

DS = datasets.MNIST(root="./mnist_data", train=True, download=False, transform=transforms.ToTensor())
DEVICE= ("cuda" if torch.cuda.is_available() else "cpu")

imgs = torch.stack([img for img,_ in DS])
mean = imgs.mean()
std = imgs.std()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean.item(),),(std.item(),))])

train_dataset = datasets.MNIST(root="./mnist_data", train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root="./mnist_data", train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3)
        self.conv2 = nn.Conv2d(8,16,3)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16*5*5,10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.fc(x)
        return(x)

model = SimpleCNN().to(DEVICE)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum =0.9)

epochs = 5

for epoch in range(epochs):
    running_loss = 0
    running_eval_loss = 0
    #train
    model.train()
    for images,labels in train_loader:
        images,labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss/len(train_loader.dataset)

    #eval
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images,labels in test_loader:
            images,labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            loss = criterion(output, labels)
            _,preds = torch.max(output,1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            running_eval_loss+= loss.item()*images.size(0)
        
        accuracy = correct/total
        eval_loss = running_eval_loss/len(test_loader.dataset)

    print(f"Epoch:{epoch} - Training Loss:{epoch_loss} | Evaluation Loss: {eval_loss} | Accuracy: {accuracy}")

torch.save(model.state_dict(), "Best_model.pth")

