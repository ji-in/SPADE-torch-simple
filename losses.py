import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generator_loss(gan_type, fake):
    loss = []
    fake_loss = 0

    for i in range(len(fake)):
        if gan_type == "gan":
            fake_loss = torch.mean(F.binary_cross_entropy_with_logits(input=torch.ones_like(fake[i][-1]), target=fake[i][-1]))
        
        loss.append(fake_loss)
    return torch.mean(torch.FloatTensor(loss))

def discriminator_loss(loss_func, real, fake):
    loss = 0
    real_loss = 0
    fake_loss = 0

    for i in range(len(fake)):
        if loss_func == 'gan':
            real_loss = torch.mean(
                F.binary_cross_entropy_with_logits(input=torch.ones_like(real[i][-1]), target=real[i][-1]))
            fake_loss = torch.mean(
                F.binary_cross_entropy_with_logits(input=torch.zeros_like(fake[i][-1]), target=fake[i][-1]))

        # loss.append(real_loss + (fake_loss))
        loss += (real_loss + fake_loss)
    # return torch.mean(torch.FloatTensor(loss))
    return loss / len(fake)

# KL loss 는 encoder 를 사용할 때에만 쓰인다.
def kl_loss(mean, logvar):
    # loss = 0.5 * torch.sum(torch.square(mean) + torch.exp(logvar) - 1 - logvar) # tensorflow 버전
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) # official 버전

def L1_loss(x, y):
    loss = torch.mean(torch.abs(x - y))

    return loss

def feature_loss(real, fake):
    loss = []

    for i in range(len(fake)):
        intermediate_loss = 0
        for j in range(len(fake[i]) - 1) :
            intermediate_loss += L1_loss(real[i][j], fake[i][j])
        loss.append(intermediate_loss)

    return torch.mean(torch.Tensor(loss))

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

