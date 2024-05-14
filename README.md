# DCGAN trained on FASHION-MNIST

This repository contains code for conducting various experiments with GANs on the Fashion-MNIST dataset. The experiments involve altering the model architecture, hyperparameters, and training strategies to assess their impact on performance.

## 1. Introduction and Background

### 1.1 Backbone Choice

The initial model exhibited poor performance, leading to the use of DC-GAN instead, leveraging Convolutional Neural Networks for efficient image processing.

### 1.2 Evaluation Metrics

The Fréchet Inception Distance (FID) was chosen as the primary evaluation metric. FID measures the distance between feature vectors of real and generated images, with lower scores indicating better quality.

## 2. Installation

### 2.1 Download the Dataset

The dataset `IMDB_Dataset.csv` is included in the repository.

### 2.2 Installing the Packages

Install the required packages using:

```bash
pip install -r requirements.txt
```

## 3. Running the Script

To run the script, execute the following command:

```bash
python main.py
```

This script will:
1. Set the device to GPU if available.
2. Load and preprocess the IMDB dataset.
3. Perform hyperparameter tuning by varying configurations of frozen encoder layers, embeddings, and pooler layers.
4. Record and save the validation loss and accuracy for each configuration to `grid_search_result.json`.

## 4. Summary of Experiments

### Experiment 1: Latent Size, d_steps, and Learning Rate Optimization
- **Parameters**: Latent Size [64, 128], d_steps [1, 2], Learning Rate [0.0002, 0.0004]
- **Best Configuration**: Latent Size 128, d_steps 1, Learning Rate 0.0004

### Experiment 2: Batch Size Impact with d_steps and LR Tuning
- **Parameters**: Batch Size [256, 128, 64], d_steps [2, 3], Learning Rate [0.0003, 0.0004, 0.0005]
- **Best Configuration**: Batch Size 256, d_steps 3, Learning Rate 0.0003

### Experiment 3: Activation Function Variation
- **Adjustment**: Changed discriminator activation to Leaky ReLU
- **Outcome**: No significant performance improvement

### Experiment 4: Loss Function Exploration
- **Adjustment**: Changed generator's loss function
- **Outcome**: Led to vanishing gradients and poor performance

## 5. Conclusion

The best performance was achieved with a batch size of 256, d_steps of 3, and a discriminator learning rate of 0.0003, yielding an FID score of 62.9709. Future work could explore beyond the tested parameter ranges for further optimization.
