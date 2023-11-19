import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm
from utils import denorm
import matplotlib.pyplot as plt
from utils import GAN_evaluator
from datetime import datetime
import logging
from scheduler import GapAwareScheduler



def train_discriminator(D, G, images, d_optimizer, criterion, hyperparams, device):
    batch_size = images.size(0)
    latent_size = hyperparams['latent_size']
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
    # Second term of the loss is always zero since real_labels == 1
    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs
        
    # Compute BCELoss using fake images
    # First term of the loss is always zero since fake_labels == 0
    #z = torch.randn(batch_size, latent_size).to(device)
    z = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs
        
    # Backprop and optimize
    d_loss = d_loss_real + d_loss_fake
    d_optimizer.zero_grad()
    # calculates gradient
    d_loss.backward()
    # Update parameters
    d_optimizer.step()
    return d_loss, real_score, fake_score

def train_generator(D, G, g_optimizer, criterion, hyperparams, device):
    batch_size, latent_size = hyperparams['batch_size'], hyperparams['latent_size']
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    # Compute loss with fake images
    #z = torch.randn(batch_size, latent_size).to(device)
    z = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = G(z)
    outputs = D(fake_images)
        
    # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
    # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
    #g_loss = criterion(outputs, real_labels)
    g_loss = - criterion(outputs, fake_labels)
    # Backprop and optimize
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images
        


def train_gan(D, G, d_optimizer, g_optimizer, criterion, data_loader, hyperparams, device, history_best_result):
    sample_dir, num_epochs = \
        hyperparams['sample_dir'], hyperparams['num_epochs']
    # Create a directory if not exists
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Initialize a variable to track the minimum generator loss
    min_fid_score = float('inf')
    Evaluator = GAN_evaluator(device)
    # Lists to store loss values
    d_losses = []
    g_losses = []
    is_scores = []
    fid_scores = []

    #scheduler = GapAwareScheduler(d_optimizer)

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

        for i, (images, _) in loop:
            images = images.to(device)
            
            # Train the discriminator
            for _ in range(hyperparams['d_steps']):
                d_loss, real_score, fake_score = train_discriminator(D, G, images, d_optimizer, criterion, hyperparams, device)

            #scheduler.update_loss(d_loss)
            # Adjust learning rate
            #scheduler.step()
            #current_lr = scheduler.optimizer.param_groups[0]['lr']
            #print(f" Current LR = {current_lr}")
            for _ in range(hyperparams['g_steps']):
                g_loss, fake_images = train_generator(D, G, g_optimizer, criterion, hyperparams, device)
            # Append losses for plotting
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
        # Print D(x) and D(G(x))
        print(f'D(x): {real_score.mean().item():.4f}, D(G(x)): {fake_score.mean().item():.4f}')
        # Evaluating the model using inception score and FID after each epoch
        inception_score = Evaluator.inception_score(fake_images)
        is_scores.append(inception_score)
        fid_score = Evaluator.calculate_fid(images, fake_images)
        fid_scores.append(fid_score)
        print(f"Inception Score: {inception_score}")
        print(f"FID Score: {fid_score}")
        logging.info(f"epoch: {epoch}, D(x): {real_score.mean().item():.4f}, "
             f"D(G(x)): {fake_score.mean().item():.4f}, "
             f"Inception Score: {inception_score}, "
             f"FID Score: {fid_score}")
        # Save real images in the first epoch
        if (epoch+1) == 1:
            save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
        
        # Save sampled images
        save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

        # Save models if current generator loss is lower than the minimum recorded generator loss
        if fid_score < min_fid_score:
            min_fid_score = fid_score
            if min_fid_score < history_best_result:
                history_best_result = min_fid_score
                # Directory for best samples
                best_sample_dir = 'best_sample'
                if not os.path.exists(best_sample_dir):
                    os.makedirs(best_sample_dir)
                best_image_path = os.path.join(best_sample_dir, 'best_image.png')
                save_image(denorm(fake_images), best_image_path)
            torch.save(G.state_dict(), 'G.ckpt')
            torch.save(D.state_dict(), 'D.ckpt')
    try:
        # Creating a folder if it doesn't exist
        folder_name = 'training_process'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_training_plot.png"
        file_path = os.path.join(folder_name, unique_name)
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # Save the plot instead of showing it
        plt.savefig(file_path)
        print(f"Plot saved successfully at {file_path}")
        plt.close()

        # Now plotting the FID scores
        unique_name_fid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_fid_plot.png"
        file_path_fid = os.path.join(folder_name, unique_name_fid)
        
        plt.figure(figsize=(10, 5))
        plt.title("FID Score During Training")
        
        # If the FID scores start from a very large value, we can apply a log scale or
        # plot the inverse of the FID scores to reduce the steepness
        plt.plot([1 / f if f != 0 else 0 for f in fid_scores], label="1/FID")  # Plot 1/FID for a different view
        
        plt.xlabel("Iterations")
        plt.ylabel("1/FID (for scaling purposes)")
        plt.legend()
        # Save the FID plot instead of showing it
        plt.savefig(file_path_fid)
        plt.close()  # Close the plot to avoid displaying it in the notebook
        print(f"FID plot saved successfully at {file_path_fid}")

        
    except Exception as e:
        print(f"An error occurred while plotting: {e}")
    return min_fid_score

