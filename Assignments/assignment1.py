import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = (15, 8)


###########################################################################
#Assignment 1
def gradient_wrt_m_and_c(inputs, labels, m, c, k):
    
    # gradient w.r.t to m is g_m 
    # gradient w.r.t to c is g_c

    x = inputs
    y = labels

    g_m = -2 * torch.sum(x[k] * (y[k] - (m * x[k]) - c))
    g_c = -2 * torch.sum(y[k] - (m*x[k]) - c)

    return g_m, g_c

# Generating y = mx + c + random noise
num_data = 1000

# True values of m and c
m_line = 3.3
c_line = 5.3

# input (Generate random data between [-5,5])
x = 10 * torch.rand(num_data) - 5

# Output (Generate data assuming y = mx + c + noise)
y_label = m_line * x + c_line + torch.randn_like(x)
y = m_line * x + c_line

# Plot the generated data points 
plt.plot(x, y_label, '.', color='g', label="Data points")
plt.plot(x, y, color='b', label='y = mx + c', linewidth=3)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()

#try out
X = torch.tensor([-0.0374,  2.6822, -4.1152])
Y = torch.tensor([ 5.1765, 14.1513, -8.2802])
m = 2
c = 3
k = torch.tensor([0, 2])

gm, gc = gradient_wrt_m_and_c(X, Y, m, c, k)

print('Gradient of m : {0:.2f}'.format(gm))
print('Gradient of c : {0:.2f}'.format(gc))    

def update_m_and_c(m, c, g_m, g_c, lr):
    updated_m = m - lr * g_m
    updated_c = c - lr * g_c

    return updated_m, updated_c

m = 2
c = 3
g_m = -24.93
g_c = 1.60
lr = 0.001
m, c = update_m_and_c(m, c, g_m, g_c, lr)

print('Updated m: {0:.2f}'.format(m))
print('Updated c: {0:.2f}'.format(c))

#-------------------------------------------------------------
# Stochastic Gradient Descent with Minibatch

# input 
X = x

# output/label
Y = y_label

num_iter = 1000
batch_size = 10

# display updated values after every 10 iterations
display_count = 20
# 

lr = 0.001
m = 2
c = 1
print()
loss = []

for i in range(0, num_iter):

    # Randomly select a training data point
    k = torch.randint(0, len(Y)-1, (batch_size,))
  
    # Calculate gradient of m and c using a mini-batch
    g_m, g_c = gradient_wrt_m_and_c(X, Y, m, c, k)
    
    # update m and c parameters
    m, c = update_m_and_c(m, c, g_m, g_c, lr)
    
    # Calculate Error
    e = Y - m * X - c
    # Compute Loss Function
    current_loss = torch.sum(torch.mul(e,e))
    loss.append(current_loss)

    if i % display_count==0:
        print('Iteration: {}, Loss: {}, updated m: {:.3f}, updated c: {:.3f}'.format(i, loss[i], m, c))
        y_pred = m * X + c
        # Plot the line corresponding to the learned m and c
        plt.plot(x, y_label, '.', color='g')
        plt.plot(x, y, color='b', label='Line corresponding to m={0:.2f}, c={1:.2f}'.
                 format(m_line, c_line), linewidth=3)
        plt.plot(X, y_pred, color='r', label='Line corresponding to m_learned={0:.2f}, c_learned={1:.2f}'.
                 format(m, c), linewidth=3)
        plt.title("Iteration : {}".format(i))
        plt.legend()

        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()
print('Loss of after last batch: {}'.format(loss[-1]))
print('Leaned "m" value: {}'.format( m))
print('Leaned "c" value: {}'.format( c))

# Plot loss vs m  
plt.figure
plt.plot(range(len(loss)),loss)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()       

# Calculate the predicted y values using the learned m and c
y_pred = m * X + c
# Plot the line corresponding to the learned m and c
plt.plot(x, y_label, '.', color='g', label='X and Y')
plt.plot(x, y, color='b', label='Line corresponding to m={0:.2f}, c={1:.2f}'.format(m_line, c_line), linewidth=3)
plt.plot(X, y_pred, color='r', label='Line corresponding to m_learned={0:.2f}, c_learned={1:.2f}'.format(m, c), linewidth=3)
plt.legend()

plt.ylabel('y')
plt.xlabel('x')
plt.show()

#####################################################################################

# random manual seed for consistency
# read more at https://pytorch.org/docs/stable/notes/randomness.html

torch.manual_seed(0)

num_data = 3

# Input 
x = 10 * torch.rand(num_data)
print(x)

# Output
y = x + torch.randn_like(x)
print(y)

###############################################################
# Linear model 
# y = mx 

# Minimum value of m
min_val = 0.0 

# Maximum value of m
max_val = 2.0 

# Number of steps between min and max values
num_steps = 10

# Step size
step_size = (max_val - min_val)/(num_steps - 1)

# Space for storing all values of m
m = torch.zeros(num_steps)

# Space for storing loss corresponding 
# to different values of m. 
loss = torch.zeros(num_steps)

# Calculate loss for all possible m
for i in range(0, num_steps):
    m[i] = min_val +  i * step_size
    e = y - m[i] * x
    loss[i] = torch.sum(torch.mul(e,e)) 


# Find the index for lowest loss
i = torch.argmin(loss)

# Minimum loss
print('Minimum Loss : ', loss[i])

# Find the value of m corresponding to lowest loss
print('Best parameter : ', m[i])

# Plot loss vs m  
plt.figure
plt.plot(m.numpy(), loss.numpy(), marker=".", markersize=14)
plt.xlabel('m')
plt.ylabel('loss')
plt.show()

##########################################################################
#Gradient Descent
num_iter = 10

lr = 0.01
m = 2 

loss = torch.zeros(num_iter)
# Calculate loss for all possible m
for i in range(0, num_iter):
    g = -2 * torch.sum(x * (y - m * x))
    m = m -  lr * g
    e = y - m * x
    loss[i] = torch.sum(torch.mul(e,e)) 
    
print('Minimum loss : ', loss[-1])
print('Best parameter : ', m)

# Plot loss vs m  
plt.figure
plt.plot(loss.numpy(), marker=".", markersize=16)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()
#####################################################################################
#Stochastic Gradient Descent
num_iter = 20

lr = 0.01
m = 2 
print()
loss = torch.zeros(num_iter)

for i in range(0, num_iter):

    # Randomly select a training data point
    k = torch.randint(0, len(y), (1,))[0]

    # Calculate gradient using a single data point
    g = -2 * x[k] * (y[k] - m * x[k])
    m = m -  lr * g
    e = y - m * x
    loss[i] = torch.sum(torch.mul(e,e)) 

print('Minimum loss : ', loss[-1])
print('Best parameter : ', m)

# Plot loss vs m  
plt.figure
plt.plot(loss.numpy(), marker=".", markersize=14)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()

###############################################################
#Stochastic Gradient Descent with Minibatch
num_data = 1000
batch_size = 10

# Input 
x = 10 * torch.rand(num_data)

# Output
y = x + torch.randn_like(x)

num_iter = 30

lr = 0.001
m = 2 
print()
loss = torch.zeros(num_iter)

for i in range(0, num_iter):

    # Randomly select a training data point
    k = torch.randint(0, len(y)-1, (batch_size,))
  
    # Calculate gradient using a mini-batch
    g = -2 * torch.sum(x[k] * (y[k] - m * x[k]))
    m = m -  lr * g
    e = y - m * x
    loss[i] = torch.sum(torch.mul(e,e)) 

print('Minimum loss : ', loss[-1])
print('Best parameter : ', m)

# Plot loss vs m  
plt.figure
plt.plot(loss.numpy())
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()

