import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from utils import set_hparams, log_hyperparameters
#from models.plain_gan import Discriminator, Generator
from models.DC_gan import DC_Generator, DC_Discriminator
from data_loader import load_fashion_mnist
from running_models import train_gan
import logging
import json
import copy


if __name__ == "__main__":
    # create logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training_log.log')
    # Check if GPU 1 is available
    device = torch.device('cuda:1')
    hyperparams = set_hparams()
    log_hyperparameters(hyperparams)
    # unpack the hyperparameters
    latent_size, hidden_size, image_size, num_epochs, batch_size, sample_dir = \
        hyperparams['latent_size'], hyperparams['hidden_size'], hyperparams['image_size'],\
        hyperparams['num_epochs'], hyperparams['batch_size'], hyperparams['sample_dir']
    train_loader, test_loader = load_fashion_mnist(batch_size)
    records = []
    history_best_result = float('inf')
    for batch_size in [256]:
        for discriminator_lr in [0.0004]:
            for d_step in [2]:
                hyperparams['batch_size'] = batch_size
                hyperparams['discriminator_lr'] = discriminator_lr
                hyperparams['d_steps'] = d_step
                #D = Discriminator(image_size, hidden_size, leaky=0.2).to(device)
                D = DC_Discriminator().to(device)
                #G = Generator(latent_size, hidden_size, image_size).to(device)
                G = DC_Generator(latent_size = latent_size).to(device)
                # Binary cross entropy loss and optimizer
                criterion = nn.BCELoss()
                #criterion = nn.BCEWithLogitsLoss()
                # initially, lr = 0.0002
                d_optimizer = torch.optim.Adam(D.parameters(), lr = discriminator_lr, betas=(0.5, 0.999))
                g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

                results = \
                    train_gan(D, G, d_optimizer, g_optimizer, criterion, train_loader, hyperparams, device, history_best_result)
                if results < history_best_result:
                    history_best_result = results

                hyperparams['results'] = results
                records.append(copy.deepcopy(hyperparams))

    print(records)
    print(f"best result: {history_best_result}")
    for record in records:
        record['results'] = float(record['results'])  # Convert to native Python float
    with open('results.json', 'w') as file:
        json.dump(records, file)
