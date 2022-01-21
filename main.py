import argparse
from dataset import load_dataset
from nets import *
# import torch.nn.functional as F
from losses import *
# import GPUtil
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(next(conv_128.parameters()).device)

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser(description='parameters for model')

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--segmap_ch', type=int, default=3, help='The size of segmap channel')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')

    # parser.add_argument('--random_style', type=bool, default=False, help='enable training with an image encoder.')
    parser.add_argument('--num_upsampling_layers', type=str, default='more',
                        choices=('normal', 'more', 'most'),
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more upsampling + resnet layer at the end of the generator")
    
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')

    parser.add_argument('--n_scale', type=int, default=2, help='number of scales')
    parser.add_argument('--n_dis', type=int, default=4, help='The number of discriminator layer')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')

    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--vgg_weight', type=int, default=10, help='Weight about perceptual loss')
    parser.add_argument('--feature_weight', type=int, default=10, help='Weight about discriminator feature matching loss')
    parser.add_argument('--kl_weight', type=float, default=0.05, help='Weight about kl-divergence')
    parser.add_argument('--gan_type', type=str, default='gan', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--TTUR', type=bool, default=True, help='Use TTUR training scheme')

    parser.add_argument('--decay_flag', type=bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')

    parser.add_argument('--display_step', type=int, default=20, help='size of results is display_step * display_step')

    return parser.parse_args()

def imsave(input, name):
    input = input.detach()
    input = torch.squeeze(input) # batch 차원을 없애줬다.
    input = input.cpu().numpy().transpose((1, 2, 0))
    plt.imshow((input * 255).astype(np.uint8))
    plt.savefig(name)

''' 수정해야 할 것 '''

if __name__ == "__main__":
    args = parse_args()

    dataset_path = './dataset/spade_celebA'
    train_dataloader = load_dataset(args, dataset_path)

    encoder = Encoder(args).to(device).train()
    generator = Generator(args).to(device).train()
    discriminator = Discriminator(args).to(device).train()

    random_style=False
    
    if args.TTUR:
        beta1 = 0.0
        beta2 = 0.9

        g_lr = args.lr / 2
        d_lr = args.lr * 2

    else:
        beta1 = args.beta1
        beta2 = asgs.beta2
        g_lr = args.lr
        d_lr = args.lr

    G_params = list(generator.parameters())
    if not random_style:
        G_params += list(encoder.parameters())
    D_params = list(discriminator.parameters())
    
    G_optim = torch.optim.Adam(G_params, lr=g_lr, betas=(beta1, beta2), weight_decay=0.1)
    D_optim = torch.optim.Adam(D_params, lr=d_lr, betas=(beta1, beta2), weight_decay=0.1)

    start_time = time.time() 
    init_lr = args.lr

    for epoch in range(args.epoch):
        # GPUtil.showUtilization()
        for i, (real_x, segmap, segmap_onehot) in tqdm(enumerate(train_dataloader)):
            
            real_x, segmap_onehot = real_x.to(device), segmap_onehot.to(device)
            if not random_style:
                x_mean, x_var = encoder(real_x)
                
            fake_x = generator(segmap_onehot, x_mean, x_var, random_style)
            random_fake_x = generator(segmap_onehot, random_style=True)
            
            real_logit = discriminator(segmap_onehot, real_x)
            fake_logit = discriminator(segmap_onehot, fake_x.detach())

            GP = 0
            
            ''' Update Discriminator '''
            D_optim.zero_grad()

            d_adv_loss = args.adv_weight * (discriminator_loss("gan", real_logit, fake_logit) + GP) # tensor(1.8223)
            d_loss = d_adv_loss

            d_loss.backward()
            D_optim.step()
             
            ''' Update Generator '''
            if i % args.n_critic == 0:
                G_optim.zero_grad()
                
                g_adv_loss = args.adv_weight * generator_loss("gan", fake_logit) # tensor(1.4685)
                g_kl_loss = args.kl_weight * kl_loss(x_mean, x_var) # tensor(1.5650, device='cuda:0', grad_fn=<MulBackward0>)
                g_vgg_loss = args.vgg_weight * VGGLoss()(real_x, fake_x) # tensor(9.3502, device='cuda:0') # 여기 fake_x.datach() 해야 하나..?
                g_feature_loss = args.feature_weight * feature_loss(real_logit, fake_logit) # tensor(26.8884)
                g_loss = (g_adv_loss + g_kl_loss + g_vgg_loss + g_feature_loss) / 4 # 수정함
                # print(g_adv_loss, g_kl_loss, g_vgg_loss, g_feature_loss)
                # print(g_loss)

                g_loss.backward()
                G_optim.step()

            ''' Update Learning Rate '''
            if args.decay_flag:
                lr = init_lr if epoch < args.decay_epoch else init_lr * (args.epoch - epoch) / (args.epoch - args.decay_epoch)

                for param_group in D_optim.param_groups:
                    param_group['lr'] = lr

                for param_group in G_optim.param_groups:
                    param_group['lr'] = lr

            ''' Visualization '''
            if i % args.display_step == 0 and i > 0:
                print(f"[Visualization!] Iteration : {i} || Generator loss: {g_loss} || discriminator loss: {d_loss}")

                imsave(real_x, f"./sample/real_{epoch}_epoch_{i}_iter.png")
                imsave(segmap, f"./sample/segmap_{epoch}_epoch_{i}_iter.png")
                imsave(fake_x, f"./sample/fake_{epoch}_epoch_{i}_iter.png")
                imsave(random_fake_x, f"./sample/random_fake_{epoch}_epoch_{i}_iter.png")
                
        ''' Save model '''
        if epoch % 10 == 0:
            # test에는 generator만 필요하니까 generator만 저장하면 되려나.
            torch.save(generator.state_dict(), f"./checkpoint/generator_weight_{epoch}epoch_{i}iter.pt")

        print(f"Epoch : {epoch} || time : {time.time() - start_time} || discriminator loss: {d_loss} || generator loss : {g_loss}")


