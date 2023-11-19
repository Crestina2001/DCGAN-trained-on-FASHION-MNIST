import torch
import torchvision
import torch.nn as nn
class DC_Generator(nn.Module):
    def __init__(self, latent_size=64, ngf=64, activation_func='relu'):
        super(DC_Generator, self).__init__()

        # Determine the activation function
        if activation_func == 'relu':
            self.activation = nn.ReLU(True)
        elif activation_func == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError("Unsupported activation function")

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            self.activation,
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),  # Adjusted kernel size and stride
            nn.BatchNorm2d(ngf * 4),
            self.activation,
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            self.activation,
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 0, bias=False),  # Adjusted kernel size, stride and padding
            nn.BatchNorm2d(ngf),
            self.activation,
            # state size. (ngf) x 28 x 28
            nn.ConvTranspose2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (1) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)


class DC_Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(DC_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),  # Adjusted kernel size and stride
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 7 x 7
            nn.Conv2d(ndf * 4, 1, 7, 1, 0, bias=False),  # Adjusted kernel size
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, input):
        return self.main(input)
