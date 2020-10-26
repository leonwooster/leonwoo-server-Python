import numpy as np
import pandas as pd
# importing the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# importing the module for calculating the performance metrics of the model
from sklearn import metrics

data_path = "./data/Advertising.csv" # loading the advertising dataset
data = pd.read_csv(data_path, index_col=0)
array_items = ["TV", "Radio", "Newspaper"] #creating an array list of the items
X = data[array_items] #choosing a subset of the dataset
y = data.Sales #sales

# dividing X and y into training and testing units
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linearreg = LinearRegression() #applying the linear regression model
linearreg.fit(X_train, y_train) #fitting the model to the training data
y_predict = linearreg.predict(X_test) #making predictions based on the testing unit
print(np.sqrt(metrics.mean_squared_error(y_test, y_predict))) #calculating the RMSE number
#output gives the RMSE number as 1.4046514230328955