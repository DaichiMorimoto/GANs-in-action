import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datasets import CycleGANDatasets

EPOCHS = 100
BATCH_SIZE = 64
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Layers
class Downsamp(nn.Module):

    def __init__(self, in_feat, out_feat, kernel_size=4, stride=2, padding=1, normalize=True):
        super(Downsamp, self).__init__()
        layers = [nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.LeakyReLU())
        if normalize:
            layers.append(nn.BatchNorm2d(out_feat))
        self.main = nn.ModuleList(layers)

    def forward(self, x):
        return self.main(x)

class Upsamp(nn.Module):

    def __init__(self, in_feat, out_feat, kernel_size=4, dropout_rate=0, size=2):
        super(Upsamp, self).__init__()
        layers = [nn.UpsamplingBilinear2d(size=size)]
        layers.append(nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=1))
        if dropout_rate:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.InstanceNorm2d(out_feat))
        self.main = nn.ModuleList(layers)

    def forward(self, x):
        return self.main(x)


# Cycle-GAN
class Generator(nn.Module):

    def __init__(self, img_shape=(128, 128), gf=32, channels=3, lambda_cycle=10.0, conv_opt='A2B'):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.gf = gf
        self.channels = channels
        self.lambda_cycle = lambda_cycle
        self.lambda_id = 0.9 * self.lambda_cycle
        self.conv_opt = conv_opt # A2B or B2A
        # Layers
        self.d1 = Downsamp(self.channels, self.gf)
        self.d2 = Downsamp(self.gf, self.gf*2)
        self.d3 = Downsamp(self.gf*2, self.gf*4)
        self.d4 = Downsamp(self.gf*4, self.gf*8)
        self.u1 = Upsamp(self.gf*8, self.gf*4)
        self.u2 = Upsamp(self.gf*4, self.gf*2)
        self.u3 = Upsamp(self.gf*2, self.gf)
        self.u4 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=2),
            nn.Conv2d(self.gf, self.channels, kernel_size=4, strides=1, padding=1),
            nn.Tanh()
            )

    def forward(self, img):
        d1 = self.d1(img) # 3 -> 32
        d2 = self.d2(d1) # 32 -> 64
        d3 = self.d3(d2) # 64 -> 128
        d4 = self.d4(d3) # 128 -> 256
        u1 = torch.cat(self.u1(d4), d3) # 256 -> 128
        u2 = torch.cat(self.u2(u1), d2) # 128 -> 64
        u3 = torch.cat(self.u3(u2), d1) # 64 -> 32
        out = self.main(u3) # 64 -> 3
        return out

    def sample_images(self, dataloader, size=4):
        idx = {'A2B': 0, 'B2A': 1}
        for batch in dataloader:
            imgs = batch[idx[self.conv_opt]][:size]
            gen_imgs = self.forward(imgs)
        return torch.cat(imgs, gen_imgs)

class Discriminator(nn.Module):

    def __init__(self, img_shape=(128, 128), df=64, channels=3):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.df = df
        self.channels = channels
        self.main = nn.Sequential(
            Downsamp(self.channels, self.df, normalize=False),
            Downsamp(self.df, self.df*2),
            Downsamp(self.df*2, self.df*4),
            Downsamp(self.df*4, self.df*8),
            Downsamp(self.df*8, 1, kernel_size=4, stride=1, padding=1),
            )

    def forward(self, img):
        y = self.main(img)
        return y

    def loss(self, imgs, gen_imgs, real, fake):
        d_real_loss = nn.MSELoss()(imgs, real)
        d_fake_loss = nn.MSELoss()(gen_imgs, fake)
        return 0.5 * (d_real_loss + d_fake_loss)


if __name__ == '__main__':
    if not os.path.exists('results/Cycle-GAN'):
        os.mkdir('results/Cycle-GAN')
    # Set GPU
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GAN
    G_AB = Generator(conv_opt='A2B').to(device)
    G_BA = Generator(conv_opt='B2A').to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    # Optimizer
    optimizer_G_AB = optim.Adam(G_AB.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G_BA = optim.Adam(G_BA.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # Dataloader
    dataset_train = CycleGANDatasets('./data/horse2zebra', is_train=True)
    dataset_test = CycleGANDatasets('./data/horse2zebra', is_train=False)
    dataloader_train = torch.utils.data.Dataloader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.Dataloader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # Labels
    real = torch.ones(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    fake = torch.zeros(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)

    print('Start Training!!')
    for epoch in range(EPOCHS):
        d_losses_A = []
        d_losses_B = []
        g_losses_A = []
        g_losses_B = []
        for imgs_A, imgs_B in dataloader_train:
            imgs_A.to(device)
            imgs_B.to(device)
            if imgs_A.size()[0] != BATCH_SIZE:
                break

            gen_imgs_B = G_AB(imgs_A)
            gen_imgs_A = G_BA(imgs_B)

            # Train Discriminator A
            optimizer_D_A.zero_grad()
            d_loss_A = D_A.loss(D(imgs_A), D(gen_imgs_A.detach()), real, fake)
            d_loss_A.backward()
            optimizer_D_A.step()
            d_losses_A.append(d_loss_A.cpu().detach().numpy())

            # Train Discriminator B
            optimizer_D_B.zero_grad()
            d_loss_B = D_B.loss(D(imgs_B), D(gen_imgs_B.detach()), real, fake)
            d_loss_B.backward()
            optimizer_D_B.step()
            d_losses_B.append(d_loss_B.cpu().detach().numpy())

            # Train Generator
            recon_imgs_A = G_BA(gen_imgs_B)
            recon_imgs_B = G_AB(gen_imgs_A)
            id_imgs_A = G_BA(imgs_A)
            id_imgs_B = G_AB(imgs_B)

            optimizer_G_BA.zero_grad()
            adv_loss_A = nn.MSELoss(D_A(gen_imgs_A), real) + nn.MSELoss(D_A(imgs_A), fake)
            cyc_loss_A = G_BA.lambda_cycle * nn.L1Loss()(imgs_A, recon_imgs_A)
            idn_loss_A = G_BA.lambda_id * nn.L1Loss()(imgs_A, id_imgs_A)
            g_loss_A = adv_loss_A + cyc_loss_A + idn_loss_A
            g_loss_A.backward()
            optimizer_G_BA.step()
            g_losses_A.append(g_loss_A.cpu().detach().numpy())

            optimizer_G_AB.zero_grad()
            adv_loss_B = nn.MSELoss(D_B(gen_imgs_B), real) + nn.MSELoss(D_B(imgs_B), fake)
            cyc_loss_B = G_AB.lambda_cycle * nn.L1Loss()(imgs_B, recon_imgs_B)
            idn_loss_B = G_AB.lambda_id * nn.L1Loss()(imgs_B, id_imgs_B)
            g_loss_B = adv_loss_B + cyc_loss_B + idn_loss_B
            g_loss_B.backward()
            optimizer_G_AB.step()
            g_losses_B.append(g_loss_B.cpu().detach().numpy())

        print("[Epoch %d/%d] [D_A loss: %f] [D_B loss: %f] [G_A loss: %f] [G_B loss: %f]"
            % (epoch, EPOCHS, np.average(d_losses_A), np.average(d_losses_B),
                np.average(g_losses_A), np.average(g_losses_B)))
        # Save Images
        gen_imgs_A = G_BA.sample_images(dataloader_test)
        gen_imgs_B = G_AB.sample_images(dataloader_test)
        torch.utils.save_image(torch(gen_imgs_A, gen_imgs_B),
                'results/Cycle-GAN/Cycle-GAN_images_'+str(epoch).zfill(3))
