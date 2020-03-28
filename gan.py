import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

BATCH_SIZE = 128
IMG_SHAPE = (28, 28)
Z_DIM = 100
INTERMEDIATE_DIM = 128
EPOCHS = 20000
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Generator(nn.Module):

    def __init__(self, img_shape, z_dim, intermediate_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.fc1 = nn.Linear(z_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, img_shape[0]*img_shape[1])

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z))
        x = torch.tanh(self.fc2(x))
        x = x.view(-1, self.img_shape[0], self.img_shape[1])
        return x

class Discriminator(nn.Module):

    def __init__(self, img_shape, intermediate_dim):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.fc1 = nn.Linear(img_shape[0]*img_shape[1], intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)

    def forward(self, x):
        x = x.view(-1, self.img_shape[0]*self.img_shape[1])
        y = F.leaky_relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(y))
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
    fig, axs = plt.subplots(nrow, ncol, figsize=(nrow, ncol), sharey=True, sharex=True)
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
    G = Generator(img_shape=IMG_SHAPE, z_dim=Z_DIM, intermediate_dim=INTERMEDIATE_DIM).to(device)
    D = Discriminator(img_shape=IMG_SHAPE, intermediate_dim=INTERMEDIATE_DIM).to(device)
    # Optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=1e-2)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-2)
    # dataset
    dataloader = load_mnist('./data', BATCH_SIZE, is_train=True)
    # Labels
    real = torch.ones(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    fake = torch.zeros(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    # Loss
    adv_loss = torch.nn.BCELoss()

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

        if epoch % 100 == 0:
            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCHS, np.average(d_losses), np.average(g_losses))
            )
            np_gen_imgs = 0.5 * gen_imgs.cpu().detach().numpy()[:16] + 0.5
            save_image(np_gen_imgs, "results/GAN_images_%d.png" % epoch, nrow=4, ncol=4)
