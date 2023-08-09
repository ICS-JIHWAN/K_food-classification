import os
import cv2
import glob
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

from sklearn.model_selection import train_test_split
from PIL import Image
from pandas import Series, DataFrame

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        print('=> Complete initializing weights')

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, **kwargs)
    if pretrained:
        model = models.AlexNet(num_classes)
    return model

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--root_path', type=str, default='/storage/jhchoi/kfood')
        self.parser.add_argument('--gpu_id', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=32)
        self.parser.add_argument('--num_classes', type=int, default=13)
        self.parser.add_argument('--lr', type=float, default=1e-4)

        self.opt, _ = self.parser.parse_known_args()

    def print_options(self):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

def mk_data(root_path):

    dir = glob.glob(os.path.join(root_path, '*'))
    cls = []
    X   = []
    Y   = []
    for d in dir:
        for c in os.listdir(d):
            p = os.path.join(d, c)
            data = sorted(glob.glob(os.path.join(p, '*')))[:-1]

            cls.append(c)
            X += data
            Y += [c for _ in range(len(data))]

    return X, Y, cls


class DataLoader():
    def __init__(self, X, Y, categories):

        self.X = X
        self.Y = Y
        self.categories = categories
        self.n_classes = len(self.categories)

        self.toTensor = self._toTensor()

        self.one_hot_encoding(categories)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        path = self.X[index]
        cls  = self.Y[index]

        if '.jpg' == path[-4:] or '.JPG' == path[-4:] or '.jpeg' == path[-5:] or '.gif' == path[-4:]:
            img = plt.imread(path)

            if img.shape[-1] != 3:
                img = Image.open(path).convert('RGB')
                img = np.array(img)

            label = np.zeros((self.n_classes))
            label[self.label_dict[cls]] = 1

        elif '.png' == path[-4:]:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[-1] != 3:
                img = Image.open(path).convert('RGB')
                img = np.array(img)

            label = np.zeros((self.n_classes))
            label[self.label_dict[cls]] = 1

        else:
            img = np.zeros((300, 300, 3))
            label = np.zeros((self.n_classes))
            ValueError('not found file')

        img  = self.toTensor(np.array(img))
        label = torch.tensor(label, dtype=torch.float16)

        return img, label

    def _toTensor(self):
        totensor = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((224, 224))
                                       ])
        return totensor

    def one_hot_encoding(self, classes):
        self.label_dict = dict()
        self.label_inverse_dict = dict()
        for i, c in enumerate(classes):
            self.label_dict[c] = i
            self.label_inverse_dict[i] = c


def correct_data(model, data_loader, object):
    softmax = nn.Softmax(dim=1)

    model.eval()
    label_dict = object.label_dict
    inverse_label_dict = object.label_inverse_dict

    correct_label = dict()
    num_data_each_class = dict()
    heatmap_dict = dict()
    for k in label_dict.keys():
        correct_label[k] = 0
        num_data_each_class[k] = 0
        heatmap_dict[k] = {j:0 for j in label_dict.keys()}

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            img, labels = data

            output = model(img.to(device))
            predict = softmax(output)
            predict = torch.max(predict, 1)[1]
            target = torch.max(labels, 1)[1]

            for i in range(img.size(0)):
                cls = target[i].item()
                pre = predict[i].item()
                num_data_each_class[inverse_label_dict[cls]] += 1
                heatmap_dict[inverse_label_dict[cls]][inverse_label_dict[pre]] += 1

                if cls == predict[i].item():
                    correct_label[inverse_label_dict[cls]] += 1

    mk_heatmap(heatmap_dict)
    print(num_data_each_class)
    print(correct_label)

def mk_heatmap(heatmap_dict):
    df = DataFrame(heatmap_dict)

    plt.pcolor(df, cmap='Greens')
    plt.title('heatmap validation 2', fontsize=20)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':

    # Configurations
    CFG = Config()
    CFG.print_options()
    config = CFG.opt

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(config.gpu_id))
        print('Using GPU!! ==> GPU_id : {}\n'.format(config.gpu_id))
    else:
        device = torch.device('cpu')
        print('Using CPU!!\n')

    # Data
    X, Y, categories = mk_data(config.root_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=421)

    dataset_object_train = DataLoader(X_train, Y_train, categories)
    train_loader = torch.utils.data.DataLoader(
        dataset_object_train,
        batch_size=config.batch_size,
        num_workers=4
    )

    dataset_object_test = DataLoader(X_test, Y_test, categories)
    test_loader = torch.utils.data.DataLoader(
        dataset_object_test,
        batch_size=config.batch_size,
        num_workers=4
    )

    model = resnet18(config.num_classes, pretrained=False).to(device)
    model.init_weights()

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    softmax = nn.Softmax(dim=1)

    if False:
        state_dict = torch.load('model31,0.77.pth')
        model.load_state_dict(state_dict['net'])
        model.to(device)

        optimizer.load_state_dict(state_dict['opt'])

        correct_data(model, test_loader, dataset_object_test)

    mean_loss_train = []
    mean_acc_train = []
    mean_loss_test  = []
    mean_acc_test   = []

    for curr_epoch in range(config.epochs):
        print('\033[95m' + '==== epoch {} starts ===='.format(curr_epoch + 1) + '\033[0m')

        model.train()

        loop_train = tqdm(train_loader, leave=True)
        epoch_loss_train = []
        epoch_acc_train  = []

        for batch_idx, data in enumerate(loop_train):
            img, labels = data

            output = model(img.to(device))

            loss = criterion(output, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict = softmax(output)
            predict = torch.max(predict, 1)[1]
            target  = torch.max(labels, 1)[1]
            correct = (target.to(device) == predict).sum()
            acc = correct / labels.size(0)

            epoch_loss_train.append(loss)
            epoch_acc_train.append(acc)

            loop_train.set_postfix(loss=loss.item(), acc=acc.item())

        curr_loss_train = sum(epoch_loss_train) / len(epoch_loss_train)
        curr_acc_train  = sum(epoch_acc_train)  / len(epoch_acc_train)
        print('\033[94m' + f"Mean loss was {curr_loss_train}" + '\033[0m')
        mean_loss_train.append(curr_loss_train)
        mean_acc_train.append(curr_acc_train)

        scheduler.step()

        # Validation
        model.eval()

        loop_test = tqdm(test_loader, leave=True)
        epoch_loss_test = []
        epoch_acc_test  = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loop_test):
                img, labels = data

                output = model(img.to(device))
                loss = criterion(output, labels.to(device))

                predict = softmax(output)
                predict = torch.max(predict, 1)[1]
                target  = torch.max(labels, 1)[1]
                correct = (target.to(device) == predict).sum()
                acc = correct / labels.size(0)

                epoch_loss_test.append(loss)
                epoch_acc_test.append(acc)

                loop_test.set_postfix(loss=loss.item(), acc=acc.item())

        curr_loss_test = sum(epoch_loss_test) / len(epoch_loss_test)
        curr_acc_test  = sum(epoch_acc_test) / len(epoch_acc_test)
        print(f'Test loss : {curr_loss_test:.2f}')
        print(f'Test Accuracy : {curr_acc_test:.2f}')

        mean_loss_test.append(curr_loss_test)
        mean_acc_test.append(curr_acc_test)

        if curr_acc_test > 0.7:
            state = {'net': model.state_dict(), 'opt': optimizer.state_dict()}
            torch.save(state, f'model{curr_epoch},{curr_acc_test:.2f}.pth')

    print('end train and test.')
    plt.plot(mean_loss_train, 'r', label='train')
    plt.plot(mean_loss_test, 'b', label='test')
    plt.title('train and test loss validation 2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()

