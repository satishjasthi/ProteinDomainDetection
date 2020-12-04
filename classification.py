import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy, shutil
import pandas as pd
from pathlib import Path


# create data
bucket_df = pd.read_csv('/home/satish27may/ProteinDomainDetection/data/PfamData/seq_len_0-300_and_num_samples_1000-10000_data.csv')
data_dir = Path('/home/satish27may/ProteinDomainDetection/data/Classification/ResNet18')
data_dir.mkdir(exist_ok=True, parents=True)
train_dir = data_dir/'train'
train_dir.mkdir(exist_ok=True, parents=True)
valid_dir = data_dir/'valid'
valid_dir.mkdir(exist_ok=True, parents=True)
train_records, test_records = [], []
for class_name in bucket_df['Class'].unique():
    class_df = bucket_df[bucket_df['Class']==class_name]
    num_train_samples = int(round(class_df.shape[0]*0.7))
    train_records.append(class_df.iloc[:num_train_samples,:])   
    test_records.append(class_df.iloc[num_train_samples:, :])
train_df = pd.concat(train_records,axis='rows').sample(frac=1)
valid_df = pd.concat(test_records, axis='rows').sample(frac=1)

for index, row in train_df.iterrows():
    img = Path(row['img_pth'])
    class_name = row['Class']
    class_dir = train_dir/f'{class_name}'
    if not class_dir.exists():
        class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir/f'{img.name}'
    shutil.copy(img, dest)
    
for index, row in valid_df.iterrows():
    img = Path(row['img_pth'])
    class_name = row['Class']
    class_dir = valid_dir/f'{class_name}'
    if not class_dir.exists():
        class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir/f'{img.name}'
    shutil.copy(img, dest)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = str(data_dir)
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)

# release the last res block for training
level1_modules = {}
for name, layer in model_ft.named_children():
    level1_modules[name] = layer
for name, param in level1_modules['layer4'].named_parameters():
    param.requires_grad=True
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(bucket_df['Class'].unique()))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)