import matplotlib.pyplot as plt  # one of the best graphics library for python
import matplotlib.animation as animation
plt.style.use('ggplot')

import torch
import numpy as np
import random
import time

plt.rcParams["figure.figsize"] = (8, 8)

seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#create data points
#mean = how far is it from the next data set.
#std  = how scattered of the data points, the lesser the value the more concentrate the data points.
num_data_points = 500
class_zero_points = torch.empty(num_data_points, 2).normal_(mean=2,std=0.5)
class_one_points = torch.empty(num_data_points, 2).normal_(mean=4,std=0.7)

plt.scatter(class_zero_points[:,0], class_zero_points[:,1], s=8, color='b', label='Class:0')
plt.scatter(class_one_points[:,0], class_one_points[:,1], s=8, color='r', label='Class:1')
plt.legend()
plt.xlim([-1, 8])
plt.ylim([-1, 8])
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()

#prepare data
label_zero = torch.zeros_like(class_zero_points[:,0], dtype=int)
label_one = torch.ones_like(class_one_points[:,0], dtype=int)

label = torch.cat([label_zero, label_one])
data_points = torch.cat([class_zero_points, class_one_points], dim=0)

print('Data points size: {}'.format(data_points.size()))
print('Label size: {}'.format(label.size()))

#implementing the perceptron
# Neuron: WX + B
def wx_plus_b(W, X, B):
    
    return torch.matmul(X, W) + B

# Derivative of WX + B w.r.t its input W and B
def grad_wx_plus_b(X):
    batch_size = X.size(0)
    g_w = X
    g_b = torch.ones(batch_size)
    
    return g_w, g_b

#sigmoid activation
# Plot sigmoid
plt.rcParams["figure.figsize"] = (10, 6)
z = torch.linspace(-10, 10, 1000)
y = torch.sigmoid(z)

plt.figure
plt.plot(z, y, color='b')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Activation')
plt.show()

#implement sigmoid activation
# Sigmoid
def sigmoid(x):

    return torch.sigmoid(x).squeeze()

# Derivative of sigmoid w.r.t. its input.
def grad_sigmoid(x):
    
    return sigmoid(x) * (1 - sigmoid(x))


# cross entropy plot y =1 
plt.rcParams["figure.figsize"] = (6, 6)
x = torch.linspace(0, 1, 1000)
y1 = -torch.log(x)
y0 = -torch.log(1-x)

plt.figure
plt.plot(x, y1, color='b', label="J(y') | y = 1")
plt.plot(x, y0, color='r', label="J(y') | y = 0")
plt.xlabel("y'")
plt.ylabel("J(y')")
plt.legend(loc='upper center')
plt.title('Binary Cross-Entropy')
plt.show()

#Implementing Binary Cross-Entropy
# Binary cross-entropy
def bce_loss(sigmoid_pred, g_truth):
    loss = - (1 - g_truth)* torch.log(1 - sigmoid_pred) - g_truth * torch.log(sigmoid_pred)
    return loss

# Derivative of binary cross-entropy w.r.t its input.
def grad_bce_loss(sigmoid_pred, g_truth):
    return - (g_truth * (1 / sigmoid_pred)) + ((1 - g_truth) * (1 / (1 - sigmoid_pred)))


class BinaryClassifierGraph:
    def __init__(self):
        """
        It is initializing the variable that will be updated in `forward` and `loss` function. 
        Storing these values will be used in `backward` function to get the gradient.
        """
        
        # default gardient is zero
        self.w0_grad = 0
        self.w1_grad = 0
        self.b_grad = 0
        
        self.x_in = None
        self.wx_plus_b_out = None
        self.sigmoid_out = None
        
        self.bce_loss = None
        
        self.grad_bce_loss = None
        
        self.g_truth = None
        
    def forward(self, w, x, b):
        # updated input value, it will be used in backward pass
        self.x_in = x
        self.b_in = b
        self.w_in = w
        
        # Intermediate node with the weighted sum
        self.wx_plus_b_out = wx_plus_b(w, x, b)
        
        # Output node after applying activation function
        self.sigmoid_out = sigmoid(self.wx_plus_b_out)
            
        return self.sigmoid_out
    
    def loss(self, g_truth):
        
        self.g_truth = g_truth
        
        # Compute the binary cross entropy loss
        self.bce_loss = bce_loss(self.sigmoid_out, g_truth)
        return self.bce_loss.mean()
    
    def backward(self):
        # Compute the gradients of Loss w.r.t neuron output (y')
        d_bce_loss = grad_bce_loss(self.sigmoid_out, self.g_truth)
        
        # Compute the gradients of neuron output(y') w.r.t weighted sum(z)
        d_sigmoid = grad_sigmoid(self.wx_plus_b_out)
        
        # Compute the gradients of weighted sum(z) w.r.t weights and bias
        d_w, d_b = grad_wx_plus_b(self.x_in)
        
        # Using chain rule to find overall gradient of Loss w.r.t weights and bias
        self.w0_grad = d_bce_loss * d_sigmoid * d_w[:,0]
        self.w1_grad = d_bce_loss * d_sigmoid * d_w[:,1]
        self.b_grad = d_bce_loss * d_sigmoid * d_b
        
        return
    
    def gradients(self):
        
        w_grad = torch.tensor([[self.w0_grad.mean()], [self.w1_grad.mean()]])
        b_grad = torch.tensor([self.b_grad.mean()])
        
        return w_grad, b_grad
        
def gradient_descent_update(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    
    return w, b

def train(w, b, data_points, label, epochs=100, lr=0.01, batch_size=10):
    
    bc = BinaryClassifierGraph()
    
    # for storing loss of each batch
    avg_train_loss = np.array([])
    
    updated_parms = []
    for epoch in range(epochs):
        
        # for storing loss of each batch in current epoch
        avg_loss = np.array([])
        
        num_baches = int(len(label)/batch_size)
        
        # Shuffle data and label
        shuffled_index = random.sample(range(len(label)), len(label))  
        s_data_points = data_points[shuffled_index]
        s_label = label[shuffled_index]
        
        print('\nEpoch: {}'.format(epoch+1))
        
        for batch_idx in range(num_baches):
            # get training data in batch
            start_index = batch_idx * batch_size
            end_index = (batch_idx + 1) * batch_size
            data = s_data_points[start_index:end_index]
            g_truth = s_label[start_index:end_index]
            
            # forward pass
            bc.forward(w, data, b)
            
            # Find loss
            loss = bc.loss(g_truth)
            
            # Backward will find gradient using chain rule 
            bc.backward()
            
            # Get gradients after they are updated using the backward function
            grad_w, grad_b = bc.gradients()
            
            # Update parameters using gradient descent
            w, b = gradient_descent_update(w, b, grad_w, grad_b, lr)
                
            # to show training results
            avg_loss = np.append(avg_loss, [loss])
            avg_train_loss = np.append(avg_train_loss, [loss])
            time.sleep(0.001)
            print("\rBatch: {0}/{1} | Avg Batch Loss: {2:.3} | Batch Loss: {3:.3} | Avg Train Loss:{4:.3}".
                  format(batch_idx+1, num_baches, avg_loss.mean(), loss.item(), avg_train_loss.mean()), end="")
    
        # storing parameters to show decision boundary animition
        updated_parms.append((w.data[0][0].clone(), w.data[1][0].clone(), b.data[0].clone()))
            
    return w, b, avg_train_loss, updated_parms

input_size = 2
w = torch.randn(input_size, 1)
b = torch.zeros(1)

w, b, avg_train_loss, updated_parms= train(w, b, data_points, label)
print('\nw:\n{}'.format(w))
print('\nb:\n{}'.format(b))

plt.scatter(class_zero_points[:,0], class_zero_points[:,1], s=8, color='b', label='Class:0')
plt.scatter(class_one_points[:,0], class_one_points[:,1], s=8, color='r', label='Class:1')
x1 = torch.linspace(-1, 8, 1000)
x2 = -(b.data[0] + w.data[0][0] * x1)/ w.data[1][0]
plt.plot(x1, x2, c='g', label='Learned decision boundary:\n{0:.2}x1 + {1:.2}x2 + {2:.2} = 0'.
         format(w.data[0][0], w.data[1][0], b.data[0]))
plt.legend()
plt.xlim([-1, 8])
plt.ylim([-1, 8])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plt.rcParams["figure.figsize"] = (15, 8)
plt.figure
plt.plot(range(len(avg_train_loss)), avg_train_loss)
plt.xlabel('Batch no.')
plt.ylabel('Batch Loss')
plt.show()


def accuracy(w, b, inputs, label):
    bc = BinaryClassifierGraph()
    prediction = bc.forward(w, inputs, b)
    pred = prediction >= 0.5
    pred = pred.squeeze()
    label = label.type(torch.bool)
    count = len(label)
    
    correct_pred = torch.sum(torch.eq(pred, label))
    accuracy = correct_pred * 1.0/count
    return accuracy

accuracy(w, b, data_points, label)


plt.rcParams["figure.figsize"] = (10, 10)

fig, ax = plt.subplots()

ax.set_xlim(-1, 8)
ax.set_ylim(-1, 8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')

line, = ax.plot(0, 0, color='g', label='Decision Boundary')
ax.scatter(class_zero_points[:,0], class_zero_points[:,1], s=8, color='b', label='Class:0')
ax.scatter(class_one_points[:,0], class_one_points[:,1], s=8, color='r', label='Class:1')
ax.legend(loc='upper right')

def plot_points_line(parm, line):
    x1 = torch.linspace(-1, 8, 1000)
    x2 = -(parm[2] + parm[0] * x1)/ parm[1]
    line.set_xdata(x1)
    line.set_ydata(x2)
    
    return line, 
    

line_animation = animation.FuncAnimation(fig, func=plot_points_line, frames=updated_parms, fargs=(line,),
                                         interval=200)
plt.show()