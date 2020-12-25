from google.colab import drive
  
# Accessing My Google Drive
drive.mount('/content/drive')

# Unzip dataset on runtime session

!unzip '/content/drive/MyDrive/Colab Notebooks/dataset.zip'


# PyramidNet

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
#from math import round
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.nn.init as init
from pathlib import Path

# PyramidNet


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,p = 0.1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.conv3 = nn.Conv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.p = p
    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = F.dropout(out, p=self.p, training=self.training)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.relu(out)
       #  out = F.dropout(out, p=self.p, training=self.training)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class PyramidNet(nn.Module):
        
    def __init__(self, dataset='cifar10', depth=158, alpha=56, num_classes=10, bottleneck=True):
        super(PyramidNet, self).__init__()   	
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.addrate = alpha / (3*n*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        elif dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth-2)/12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth-2)/8)

                layers[depth]= [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])

            self.inplanes = 64            
            self.addrate = alpha / (sum(layers[depth])*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
        return x


# Hyperparameters

SAVEPATH = '/content/drive/My Drive/'
WEIGHTDECAY = 5e-4 
MOMENTUM = 0.9
BATCHSIZE = 256
LR = 0.1
EPOCHS = 200
PRINTFREQ = 10

# Tensorboard

from tensorboardX import SummaryWriter
summary = SummaryWriter()

# Train Model

import time

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

random_seed = 2014102316
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministirc = True
torch.backends.cudnn.benchmark = False
def main():

    model = PyramidNet()
    
    # model.load_state_dict(torch.load(SAVEPATH+'154_58_157_model_weight.pth'))
    # model.train()
    
    # optimizer/learning rate scheduler/criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60,120,180],
                                                     gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    ###########################################################

    model = model.cuda()
    criterion = criterion.cuda()

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 2000000:
        print('Your model has the number of parameters more than 2 millions..')
        return

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        './train', transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCHSIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    
    # val_dataset = torchvision.datasets.ImageFolder('./valid', transform=valid_transform)

    last_top1_acc = 0
    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # learning rate scheduling
        scheduler.step()

        # Save model each epoch
        torch.save(model.state_dict(), SAVEPATH+'model_weight.pth')
        if epoch > 100 : 
          name = 'model_weight.pth'+ str(epoch)
          torch.save(model.state_dict(), SAVEPATH+name)
          print(name + " 저장 되었습니다.")


    print(f"Last Top-1 Accuracy: {last_top1_acc}")
    print(f"Number of parameters: {pytorch_total_params}")



def train(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    beta = 1.0
    cutmix_prob = 0.5
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        r = np.random.rand(1)
        
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)
        
        # add value to tensorboard

        # compute output

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINTFREQ == 0:
            progress.print(i)

    summary.add_scalar('loss', losses.avg, epoch)
    summary.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
    summary.add_scalar('Acc@1',top1.avg,epoch)
    print('=> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg
    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == "__main__":
    main()
    writer.close()


