import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from LogisticRegressionModel import LogisticRegression
from trainer import Trainer
from ConfusionClass import ConfusionMatrix
from OtherMatrixClass import OtherMatrix

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = (8, 8)

# for reproducible results
seed = 42  
rng = np.random.RandomState(seed)
torch.manual_seed(seed)


# generate two class classification problem
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=seed, n_clusters_per_class=1
)

# add unifom random noise
X += 4 * rng.uniform(size=X.shape)

print('Inputs (X) shape: {}'.format(X.shape))
print('Lables (y) shape: {}'.format(y.shape))

plt.scatter(X[:,0],X[:,1],c=y,edgecolor='k')
plt.show()

# Divide data into train (0.75)vand test (0.25) set. From sklearn
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# train data from numpy to torch
x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()

# create model object
log_regression = LogisticRegression(n_features=2)

# define loss. Here binary cross-entropy loss
criterion = nn.BCELoss()

# define optimizer. Here Stochastic Gradient Descent  
optimizer = torch.optim.SGD(log_regression.parameters(), lr=0.01)

# create trainer object
trainer = Trainer(log_regression, criterion, optimizer, 200)

# train the model
trainer.fit(x_train, y_train)

# test data from numpy to torch
x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()

# probability of class one prediction
y_predicted = trainer.predict(x_test)

# threshold probability 0.5

cm = ConfusionMatrix()

thres_prob = 0.5
predictions = y_predicted > thres_prob

# reset cunfusion matrix
cm.reset()

# compute confusion matrix
cm.add(predictions, y_test)


print('Confusion Matrix for threshold probability 0.5:\n{}'.format(cm.confusion_matrix()))

om = OtherMatrix()
acc = om.accuracy(cm, 0.5, y_predicted, y_test)
print('Accuracy at threhold 0.5: {}'.format(acc))

pre = om.precision(cm, 0.5, y_predicted, y_test)
print('Precision at threhold 0.5: {0:.3}'.format(pre))

rec = om.recall(cm, 0.5, y_predicted, y_test)
print('Recall at threhold 0.5: {0:.3}'.format(rec))

########################0.6
thres_prob = 0.6
predictions = y_predicted > thres_prob

# reset confusion matrix
cm.reset()

# compute confusion matrix
cm.add(predictions, y_test)
print('Confusion Matrix for threshold probability 0.6:\n{}'.format(cm.confusion_matrix()))

