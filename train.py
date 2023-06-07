import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import wandb
from sklearn.model_selection import KFold


seed_numbers = [4, 5, 6] 
lr  =  0.05
bs = 64
epochs = 50

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data transforms
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=(-15,15)),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# cutmix algorism
#[reference]] CutMix 나름대로 정리하기|작성자 퍼렁이
def cutmix(size, lam):
    W = size[2]# M x C x W x H 
    H = size[3]
    cut_rate = np.sqrt(1. - lam)
    cut_w = int(W * cut_rate)
    cut_h = int(H * cut_rate)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Define training function
def train_one_epoch(trainloader, model, optimizer, criterion, epoch): 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    beta = 1.0
    cutmix_prob = 0.5

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
        optimizer.zero_grad()

        r = np.random.rand(1)
        if beta > int(0.0) and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = cutmix(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output = model(inputs)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(inputs)
            loss = criterion(output, labels)
        outputs = model(inputs)

        # compute train accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # compute train loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 100))
            running_loss = 0.0
        
    acc = correct / total
    loss = running_loss / i
    return acc, loss

def evaluate_one_epoch(valloader, model, criterion): 
    model.eval()
    
    # validate the model
    correct = 0
    total = 0
    val_loss = 0
    itr = 0
    with torch.no_grad():
        for data in valloader:
            itr += 1
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
            outputs = model(images)
            
            # compute test accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # compute test loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    acc = correct / total
    loss = val_loss / itr

    print('Accuracy and loss of the network on the validatoin images: %f %%, %f' % (100 * acc, loss))
    return acc, loss

# Load the CIFAR-10 dataset
data_path = './data'
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
valset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

for seed_number in seed_numbers:
# fix random seed
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(trainloader, model, optimizer, criterion, epoch)
        val_acc, val_loss = evaluate_one_epoch(valloader, model, criterion)

    print('Finished Training')

    # Save the checkpoint of the last model
    PATH = '.models/resnet18_cifar10_%f_%d.pth' % (lr, seed_number)
    torch.save(model.state_dict(), PATH)
