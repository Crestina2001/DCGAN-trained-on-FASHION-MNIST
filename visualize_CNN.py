from torchviz import make_dot
from models.DC_gan import DC_Generator, DC_Discriminator
import torch

# Assume 'model' is an instance of your DC_Generator or DC_Discriminator class
dummy_input = torch.randn(1, 1, 28, 28)  # Adjust the size based on your model
model = DC_Discriminator()
output = model(dummy_input)

# Visualize the model
dot = make_dot(output, params=dict(list(model.named_parameters())))
dot.render("model_visualization", format="png")
