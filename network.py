import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
from os.path import join


import torch.nn as nn

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


__all__ = ['AlexNet256', 'alexnet256']

class AlexNet256(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(

            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet256(pretrained=False, from_numpy=False, version=1, **kwargs):


    if pretrained:
        if from_numpy:
            model = alexnet256_from_numpy()
        else:
            model = AlexNet256(**kwargs)
            model.load_state_dict(version=version, save=False,
                                  model_weights=torch.load(join("data", "pretrained", "pretrain_alexnet256_v"+str(version))))
    return model


def alexnet256_from_numpy(version=1, save=True, model_weights=join("data", "pretrained", "reference_pretrain.npy")):

    net = AlexNet256()
    net_data = dict(np.load(model_weights, encoding='bytes').item())

    perm_list = [(3, 2, 1, 0), (3, 2, 0, 1)]

    net.state_dict()['features.0.weight'].data.copy_(torch.tensor(net_data['conv1'][0]).permute(perm_list[version]))
    net.state_dict()['features.0.bias'].data.copy_(torch.tensor(net_data['conv1'][1]))
    net.state_dict()['features.4.weight'].data.copy_(torch.tensor(net_data['conv2'][0]).permute(perm_list[version]))
    net.state_dict()['features.4.bias'].data.copy_(torch.tensor(net_data['conv2'][1]))
    net.state_dict()['features.8.weight'].data.copy_(torch.tensor(net_data['conv3'][0]).permute(perm_list[version]))
    net.state_dict()['features.8.bias'].data.copy_(torch.tensor(net_data['conv3'][1]))
    net.state_dict()['features.10.weight'].data.copy_(torch.tensor(net_data['conv4'][0]).permute(perm_list[version]))
    net.state_dict()['features.10.bias'].data.copy_(torch.tensor(net_data['conv4'][1]))
    net.state_dict()['features.12.weight'].data.copy_(torch.tensor(net_data['conv5'][0]).permute(perm_list[version]))
    net.state_dict()['features.12.bias'].data.copy_(torch.tensor(net_data['conv5'][1]))

    net.state_dict()['classifier.0.weight'].data.copy_(torch.tensor(net_data['fc6'][0]).permute(1, 0))
    net.state_dict()['classifier.0.bias'].data.copy_(torch.tensor(net_data['fc6'][1]))
    net.state_dict()['classifier.3.weight'].data.copy_(torch.tensor(net_data['fc7'][0]).permute(1, 0))
    net.state_dict()['classifier.3.bias'].data.copy_(torch.tensor(net_data['fc7'][1]))

    if save: torch.save(net.state_dict(), join('data', 'pretrained', 'pretrain_alexnet256_v'+str(version)))

    return net




class AlexNetFc(nn.Module):
  def __init__(self, hash_bit, use_hashnet=True,pretrained=True, scale_tanh=True, tanh_step=200,
                     alternative_model=False, alt_version=1):
    super(AlexNetFc, self).__init__()
    if alternative_model:
        model_alexnet = alexnet256(pretrained=pretrained, version=alt_version, from_numpy=True)
    else:
        model_alexnet = models.alexnet(pretrained=pretrained)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_hashnet = use_hashnet
    self.hash_layer = nn.Linear(model_alexnet.classifier[6].in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = tanh_step
    self.gamma = 0.005
    self.power = 0.5
    self.scale_tanh = scale_tanh;
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0 and self.scale_tanh:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y;

  def output_num(self):
    return self.__in_features

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
class ResNetFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}
class VGGFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_hashnet = use_hashnet
    self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.features(x)
    x = x.view(x.size(0), 25088)
    x = self.classifier(x)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features
