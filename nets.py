import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
#                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
class Encoder(nn.Module): # 완성
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.channel = args.ch # 64
        self.img_ch, self.img_height, self.img_width = args.img_ch, args.img_height, args.img_width

        self.layer = nn.Sequential( 
            # 64
            nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel, affine=False),
            # 128
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel*2, affine=False),
            # 256
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=self.channel*2, out_channels=self.channel*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel*4, affine=False), 
            # 512
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=self.channel*4, out_channels=self.channel*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel*8, affine=False),
            # 512
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=self.channel*8, out_channels=self.channel*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel*8, affine=False),
        )
        # 512
        self.sudden_layer = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(in_channels=self.channel*8, out_channels=self.channel*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel*8, affine=False),
        )

        self.lrelu = nn.LeakyReLU(0.2, False)
        self.fc_mu = nn.Linear(in_features=self.channel * 8 * 4 * 4, out_features=256) 
        self.fc_var = nn.Linear(in_features=self.channel * 8 * 4 * 4, out_features=256) 
        
    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        
        x = self.layer(x) # (bs, 512, 1, 1)

        if self.img_height >= 256 or self.img_width >= 256: # 한번 더 레이어 통과
            x = self.sudden_layer(x)
        
        # shape of x is (bs, 512, 4, 4)
        
        x = self.lrelu(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        var = self.fc_var(x)

        return mu, var

class SPADEResBlock(nn.Module): # 완성
    def __init__(self, channel, args, in_channel):
        super(SPADEResBlock, self).__init__()
        
        self.args = args
        self.channel = channel

        self.in_channel = in_channel # in_channel = x.shape[1]
        self.mid_channel = min(self.in_channel, self.channel)
        # print(self.in_channel)

        self.spade_1 = SPADE(self.in_channel, self.args)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.mid_channel, kernel_size=3, stride=1, padding=1)

        self.spade_2 = SPADE(self.mid_channel, self.args)
        self.conv_2 = nn.Conv2d(in_channels=self.mid_channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, False)

        self.spade_shortcut = SPADE(self.in_channel, self.args)
        self.conv_shortcut = nn.Conv2d(in_channels=self.in_channel, out_channels=channel, kernel_size=1, stride=1)

    def forward(self, segmap, x_init):
        x = self.spade_1(segmap, x_init)
        x = self.lrelu(x)
        x = self.conv_1(x)

        x = self.spade_2(segmap, x)
        x = self.lrelu(x)
        x = self.conv_2(x)

        if self.in_channel != self.channel:
            x_init = self.spade_shortcut(segmap, x_init)
            x_init = self.conv_shortcut(x_init)
        
        return x + x_init

class SPADE(nn.Module): # 완성
    def __init__(self, in_channel, args):
        super(SPADE, self).__init__()
        self.img_ch = args.img_ch
        self.in_channel = in_channel
        # print(self.in_channel)

        self.instance_norm = nn.InstanceNorm2d(self.img_ch, affine=False)
        
        # self.conv_128 = nn.Conv2d(in_channels=self.img_ch, out_channels=128, kernel_size=5, stride=1, padding=2)

        self.conv_gamma = nn.Conv2d(in_channels=128, out_channels=self.in_channel, kernel_size=5, stride=1, padding=2)
        self.conv_beta = nn.Conv2d(in_channels=128, out_channels=self.in_channel, kernel_size=5, stride=1, padding=2)

        self.relu = nn.ReLU()

    def down_sample(self, x, scale_factor_w, scale_factor_h):
        _, _, h, w = x.shape
        new_size = [h // scale_factor_h, w // scale_factor_w]

        return F.interpolate(x, size=new_size, mode='nearest')

    def forward(self, segmap, x_init):
        x = self.instance_norm(x_init)

        _, _, x_h, x_w = x_init.shape
        _, _, segmap_h, segmap_w = segmap.shape

        factor_h = segmap_h // x_h
        factor_w = segmap_w // x_w

        segmap_down = self.down_sample(segmap, factor_h, factor_w)
        # print(segmap_down.shape)

        conv_128 = nn.Conv2d(in_channels=segmap_down.shape[1], out_channels=128, kernel_size=5, stride=1, padding=2)
        conv_128.to(device)
        # print(next(conv_128.parameters()).device)
        segmap_down = conv_128(segmap_down)
        segmap_down = self.relu(segmap_down)

        segmap_gamma = self.conv_gamma(segmap_down) # (bs, self.in_channel, 256, 256)
        segmap_beta = self.conv_beta(segmap_down) # (bs, self.in_channel, 256, 256)
        
        x = x * (1 + segmap_gamma) + segmap_beta
        
        return x

class Generator(nn.Module): # 완성
    def __init__(self, args): # x_mean과 x_var 은 encoder 에서 받는다.
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.ch
        self.img_ch, self.img_height, self.img_width = args.img_ch, args.img_height, args.img_width
        self.batch_size = args.batch_size
        # self.random_style = random_style # default=False
        self.num_upsampling_layers = args.num_upsampling_layers # default='more'

        self.channel = self.ch * 4 * 4 # 1024
        
        if self.num_upsampling_layers == 'normal':
            self.num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            self.num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            self.num_up_layers = 7

        self.z_height = self.img_height // (pow(2, self.num_up_layers)) # 4
        self.z_width = self.img_width // (pow(2, self.num_up_layers)) # 4
        
        # self.fc = nn.Linear(in_features=self.in_channel, out_features=self.z_width * self.z_height * self.channel) # out_features = 16384
        
        self.spade_resblock_0 = SPADEResBlock(self.channel, self.args, self.channel) # 1024
        self.spade_resblock_1 = SPADEResBlock(self.channel//2, self.args, self.channel) # 512
        self.spade_resblock_2 = SPADEResBlock(self.channel//2//2, self.args, self.channel//2) # 256
        self.spade_resblock_3 = SPADEResBlock(self.channel//2//2//2, self.args, self.channel//2//2) # 128
        self.spade_resblock_4 = SPADEResBlock(self.channel//2//2//2//2, self.args, self.channel//2//2//2) # 64
        self.spade_resblock_5 = SPADEResBlock(self.channel//2//2//2//2//2, self.args, self.channel//2//2//2//2) # 32

        self.lrelu = nn.LeakyReLU(0.2, False)
        # self.conv = nn.Conv2d(in_channels=, out_channels=self.img_ch, kernel_size=3, stride=1, padding=1) ### in_channels 채워넣기!!!
        self.tanh = nn.Tanh()

    def up_sample(self, x, scale_factor=2):
        _, _, h, w = x.shape
        return F.interpolate(x, size=(h * scale_factor, w * scale_factor), mode='bilinear')

    def z_sample(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        return mean + torch.exp(logvar * 0.5) * eps
    
    def forward(self, segmap, x_mean=None, x_var=None, random_style=False):

        if random_style:
            x = torch.randn(self.batch_size, self.ch * 4).to(device) # (bs, 256)
        else:
            x = self.z_sample(x_mean, x_var)

        fc = nn.Linear(in_features=x.shape[1], out_features=self.z_width * self.z_height * self.channel).to(device) # out_features = 16384
        x = fc(x)
        x = x.view(self.batch_size, self.channel, self.z_height, self.z_width) # (1024, 4, 4)
        x = self.spade_resblock_0(segmap, x) # 1024

        x = self.up_sample(x, scale_factor=2)
        x = self.spade_resblock_0(segmap, x) # 1024

        if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
            x = self.up_sample(x, scale_factor=2)
        
        x = self.spade_resblock_0(segmap, x) # 1024

        x = self.up_sample(x, scale_factor=2)
        x = self.spade_resblock_1(segmap, x) # 512
        
        x = self.up_sample(x, scale_factor=2)
        x = self.spade_resblock_2(segmap, x) # 256
        
        x = self.up_sample(x, scale_factor=2)
        x = self.spade_resblock_3(segmap, x) # 128
        
        x = self.up_sample(x, scale_factor=2)
        x = self.spade_resblock_4(segmap, x) # 64

        if self.num_upsampling_layers == 'most':
            x = self.upsample(x, scale_factor=2)
            x = self.spade_resblock_5(segmap, x) # 32
        
        x = self.lrelu(x)
        conv = nn.Conv2d(in_channels=x.shape[1], out_channels=self.img_ch, kernel_size=3, stride=1, padding=1).to(device)
        x = conv(x)
        x = self.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.n_scale = args.n_scale
        self.channel = args.ch # 64
        self.n_dis = args.n_dis

        self.lrelu = nn.LeakyReLU(0.2, False)

        self.conv_1 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel * 2, kernel_size=4, stride=2, padding=1) # out_channels=128
        self.conv_2 = nn.Conv2d(in_channels=self.channel * 2, out_channels=self.channel * 4, kernel_size=4, stride=2, padding=1) # out_channels=256
        self.conv_3 = nn.Conv2d(in_channels=self.channel * 4, out_channels=self.channel * 8, kernel_size=4, stride=2, padding=1) # out_channels=512
        self.conv_4 = nn.Conv2d(in_channels=self.channel * 8, out_channels=self.channel * 8, kernel_size=4, stride=1, padding=1) # out_channels=512

        self.instance_norm_1 = nn.InstanceNorm2d(self.channel * 2, affine=False)
        self.instance_norm_2 = nn.InstanceNorm2d(self.channel * 4, affine=False)
        self.instance_norm_3 = nn.InstanceNorm2d(self.channel * 8, affine=False)
        self.instance_norm_4 = nn.InstanceNorm2d(self.channel * 8, affine=False)

        self.conv_dlogit = nn.Conv2d(in_channels=self.channel * 8, out_channels=1, kernel_size=4, stride=1, padding=1)

    def forward(self, segmap, x_init):
        D_logit = []
        for scale in range(self.n_scale):
            feature_loss = []

            x = torch.cat([segmap, x_init], dim=1)
            conv_0 = nn.Conv2d(in_channels=x.shape[1], out_channels=self.channel, kernel_size=4, stride=2, padding=1).to(device)
            x = conv_0(x)
            x = self.lrelu(x)
            feature_loss.append(x)

            x = self.conv_1(x)
            x = self.instance_norm_1(x)
            x = self.lrelu(x)
            feature_loss.append(x)

            x = self.conv_2(x)
            x = self.instance_norm_2(x)
            x = self.lrelu(x)
            feature_loss.append(x)

            x = self.conv_3(x)
            x = self.instance_norm_3(x)
            x = self.lrelu(x)
            feature_loss.append(x)

            x = self.conv_4(x)
            x = self.instance_norm_4(x)
            x = self.lrelu(x)
            feature_loss.append(x)

            x = self.conv_dlogit(x)
            feature_loss.append(x)

            D_logit.append(feature_loss)

        return D_logit
