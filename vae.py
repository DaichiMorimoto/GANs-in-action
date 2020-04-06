import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary

BATCH_SIZE = 100
ORIGINAL_DIM = 784 # MNIST (784=28*28)
LATENT_DIM = 2
INTERMEDIATE_DIM = 256
EPOCHS = 20
EPSILON_STD = 1.0
IS_TRAIN = True
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AutoEncoder(nn.Module):

    def __init__(self, original_dim, latent_dim, intermediate_dim, epsilon_std):
        super(AutoEncoder, self).__init__()
        self.epsilon_std = epsilon_std
        # Encoder
        self.fc = nn.Linear(original_dim, intermediate_dim)
        self.fc_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc_log_var = nn.Linear(intermediate_dim, latent_dim)
        # Decoder
        self.fc1 = nn.Linear(latent_dim, intermediate_dim) # Decoding
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(intermediate_dim, original_dim) # Flat Decoded

    def encode(self, x):
        x = F.relu(self.fc(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

    def decode(self, z):
        y = F.relu(self.fc1(z))
        y = self.drop1(y)
        y = F.sigmoid(self.fc2(y))
        return y

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.sampling(z_mean, z_log_var)
        y = self.decode(z)
        return y, z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.size())
        sample = z_mean + torch.exp(0.5*z_log_var) * epsilon
        return sample

    def loss(self, z_mean, z_log_var, x, y):
        delta=1e-8
        KL = 0.5*torch.sum(1.0 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        recon = torch.mean(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta))
        loss = -(KL + recon)
        return loss

def load_mnist(path, batch_size, is_train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
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


if __name__ == '__main__':
    # Auto Encoder
    vae = AutoEncoder(
            original_dim=ORIGINAL_DIM,
            latent_dim=LATENT_DIM,
            intermediate_dim=INTERMEDIATE_DIM,
            epsilon_std=EPSILON_STD
            )
    # summary(vae, (1, ORIGINAL_DIM))
    if os.path.exists('./models/vae'):
        vae.load_state_dict(torch.load('./models/vae'))

    optimizer = optim.Adam(vae.parameters(), lr=1e-2)
    trainloader = load_mnist('./data', BATCH_SIZE, is_train=True)
    testloader = load_mnist('./data', BATCH_SIZE, is_train=False)

    if IS_TRAIN: # TRAIN
        print('Start Training...')
        vae.train()
        for epoch in range(EPOCHS):
            losses = []
            for i, (inputs, _) in enumerate(trainloader):
                # Forward
                outs, z_mean, z_log_var = vae(inputs)
                # Backward
                optimizer.zero_grad()
                loss = vae.loss(z_mean, z_log_var, inputs, outs)
                loss.backward()
                optimizer.step()
                # Append
                losses.append(loss.detach().numpy())
            print("EPOCH: {} loss: {}".format(epoch, np.average(losses)))
        print('Finish Trainiing!!')
        torch.save(vae.state_dict(), './models/vae')
        print('Saved Model')

    vae.eval()
    fig = plt.figure(figsize=(10, 2))
    zs = []
    for inputs, t in testloader:
        # original
        for i, im in enumerate(inputs.view(-1, 28, 28).detach().numpy()[:10]):
          ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
          ax.imshow(im, 'gray')

        # inputs = inputs.to(device)
        outs, z_mean, z_log_var = vae(inputs)
        z = vae.sampling(z_mean, z_log_var)
        zs.append(z)
        for i, im in enumerate(outs.view(-1, 28, 28).detach().numpy()[:10]):
          ax = fig.add_subplot(2, 10, i+11, xticks=[], yticks=[])
          ax.imshow(im, 'gray')

        print(inputs.detach().numpy()[0], outs.view(-1, 28, 28).detach().numpy()[0])
        break

    fig.savefig('results/autoencoder.png')
