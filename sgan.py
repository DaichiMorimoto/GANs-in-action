import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

Z_DIM = 100 # 100, 1, 1
EPOCHS = 5000
BATCH_SIZE = 32
NUM_CLASSES = 10
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # Network
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
            # 28*28*2 -> 14*14*64
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            # 14*14*64 -> 7*7*64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # 7*7*64 -> 3*3*128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5), # NEW!!

            nn.Flatten(),
            nn.Linear(3*3*128, NUM_CLASSES), # NEW!!
        )
        self.supervised_layer = nn.Softmax()

    def unsupervised_layer(self, x):
        return 1.0 - (1.0 / (torch.sum(torch.exp(x)) + 1.0))

    def forward(self, x):
        y = self.main(x)
        return self.supervised_layer(y), self.unsupervised_layer(y)

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
    if not os.path.exists('results/SGAN'):
        os.mkdir('results/SGAN')
    # Set GPU
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GAN
    G = Generator(z_dim=Z_DIM).to(device)
    D = Discriminator().to(device)
    # Optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=1e-3)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-3)
    # Dataloader
    train_labeled_dataloader = load_mnist('./data', BATCH_SIZE, is_train=True)
    train_unlabeled_dataloader = load_mnist('./data', BATCH_SIZE, is_train=True)
    test_dataloader = load_mnist('./data', BATCH_SIZE, is_train=False)
    # Labels
    real = torch.ones(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    fake = torch.zeros(BATCH_SIZE, 1, device=device, dtype=dtype, requires_grad=False)
    # Loss
    adv_loss = torch.nn.BCELoss()
    aux_loss = torch.nn.CrossEntropyLoss()

    print('Start Training!!')
    for epoch in range(EPOCHS):
        d_aux_losses = []
        d_adv_losses = []
        g_losses = []
        for i, (labeled_imgs, labels) in enumerate(train_labeled_dataloader):
            if i > 3:
                break
            # DataLoader Unlabeled
            idx = np.random.randint(0, NUM_CLASSES)
            unlabeled_imgs, _ = train_unlabeled_dataloader[idx]
            unlabeled_imgs.to(device)

            # DataLoader Labeled
            tensor_labels = torch.eye(NUM_CLASSES)[labels]
            labeled_imgs.to(device)
            if labeled_imgs.size()[0] != BATCH_SIZE or unlabeled_imgs.size()[0] != BATCH_SIZE:
                break

            # Generate Images
            z = torch.randn(BATCH_SIZE, Z_DIM, device=device, dtype=dtype)
            gen_imgs = G(z)

            # Discriminator
            _, d_real_labels = D(labeled_imgs)
            d_real, _ = D(unlabeled_imgs)
            d_fake, _ = D(gen_imgs.detach())

            # Train Discriminator supervised
            optimizer_D.zero_grad()
            d_aux_loss = aux_loss(d_real_labels, tensor_labels)
            d_aux_loss.backward()
            optimizer_D.step()
            d_aux_losses.append(d_aux_loss.cpu().detach().numpy())

            # Train Discriminator unsupervised
            optimizer_D.zero_grad()
            d_adv_loss = 0.5 * (adv_loss(d_real, real) + adv_loss(d_fake, fake))
            d_adv_loss.backward()
            optimizer_D.step()
            d_adv_losses.append(d_adv_loss.cpu().detach().numpy())

            # Train Generator
            optimizer_G.zero_grad()
            g_valid, _ = D(gen_imgs)
            g_loss = adv_loss(g_valid, real)
            g_loss.backward()
            optimizer_G.step()
            g_losses.append(g_loss.cpu().detach().numpy())

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCHS, np.average(d_losses), np.average(g_losses))
        )
        np_gen_imgs = gen_imgs.cpu().detach().numpy[:16]
        save_image(np_gen_imgs, "results/SGAN/SGAN_images_%d.png" % epoch, nrow=4, ncol=4)
