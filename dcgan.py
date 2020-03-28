import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

Z_DIM = 100 # 100, 1, 1
EPOCHS = 50
BATCH_SIZE = 128
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 256*7*7)
        self.main = nn.Sequential(
            # 7*7 -> 14*14
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # 14*14 -> 14*14
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # 14*14 -> 28*28
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 7, 7)
        x = self.main(x)
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 28*28 -> 14*14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            # 14*14 -> 7*7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # 7*7 -> 3*3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(3*3*128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.main(x)
        return y


def load_mnist(path, batch_size, is_train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
        ])
    dataset = datasets.MNIST(
            root=path,
            train=is_train,
            download=True,
            transform=transform
            )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
            )
    return dataloader

def save_image(np_gen_imgs, img_path, nrow=4, ncol=4):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow), sharey=True, sharex=True)
    cnt = 0
    for i in range(nrow):
        for j in range(ncol):
            axs[i, j].imshow(np_gen_imgs[cnt], 'gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(img_path)

if __name__ == '__main__':
    # Set GPU
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GAN
    G = Generator(z_dim=Z_DIM).to(device)
    D = Discriminator().to(device)
    # Optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=1e-3)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-3)
    # dataset
    dataloader = load_mnist('./data', BATCH_SIZE, is_train=True)
    # Labels
    real = torch.ones(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    fake = torch.zeros(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    # Loss
    adv_loss = torch.nn.BCELoss()

    print('Start Training!!')
    for epoch in range(EPOCHS):
        d_losses = []
        g_losses = []
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs.to(device)
            if real_imgs.size()[0] != BATCH_SIZE:
                break

            z = torch.randn(BATCH_SIZE, Z_DIM, device=device, dtype=dtype)
            gen_imgs = G(z)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adv_loss(D(real_imgs), real)
            fake_loss = adv_loss(D(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()
            d_losses.append(d_loss.cpu().detach().numpy())

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = adv_loss(D(gen_imgs), real)
            g_loss.backward()
            optimizer_G.step()
            g_losses.append(g_loss.cpu().detach().numpy())

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCHS, np.average(d_losses), np.average(g_losses))
        )
        np_gen_imgs = 0.5 * gen_imgs.view(-1, 28, 28).cpu().detach().numpy()[:16] + 0.5
        save_image(np_gen_imgs, "results/DCGAN_images_%d.png" % epoch, nrow=4, ncol=4)
