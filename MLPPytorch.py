import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Get reproducible results
torch.manual_seed(0)

# Define the model
class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden_layer_nodes, num_outputs):
        # Initialize super class
        super().__init__()

        # Add hidden layer 
        self.linear1 = nn.Linear(num_inputs, num_hidden_layer_nodes)

        # Add output layer
        self.linear2 = nn.Linear(num_hidden_layer_nodes, num_outputs)
        

    def forward(self, x):
        # Forward pass through hidden layer with 
        x = F.relu(self.linear1(x))
        
        # Foward pass to output layer
        return self.linear2(x)

# Num data points
num_data = 1000

# Network parameters
num_inputs = 1000
num_hidden_layer_nodes = 100
num_outputs = 10

# Training parameters
num_epochs = 200

# Create random Tensors to hold inputs and outputs
x = torch.randn(num_data, num_inputs)
y = torch.randn(num_data, num_outputs)

# Construct our model by instantiating the class defined above
model = MLP(num_inputs, num_hidden_layer_nodes, num_outputs)

# Define loss function
loss_function = nn.MSELoss(reduction='sum')

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4)


for t in range(num_epochs):

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = loss_function(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # Calculate gradient using backward pass
    loss.backward()

    # Update model parameters (weights)
    optimizer.step()