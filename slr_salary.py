import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
salary = pd.read_csv(r"C:\Users\vishw\Downloads\Salary_Data (1)\Salary_Data.csv")

print("dataset shape:", salary.shape)

# Define the target variable y and feature variable x
y = salary.iloc[:, 1]
x = salary.iloc[:, 0]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

# Reshape x_train and x_test to be 2D arrays (as required by the model)
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# Create the linear regression model and train it
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(x_test)

# Plot the results
plt.scatter(x_test, y_test, color='red')  # Plot actual data points
plt.plot(x_test, y_pred, color='blue')  # Plot predicted line
plt.show()

#slope y=mx+c
#m=coeffecient

coeff=regressor.coef_
#intercept=c

interce=regressor.intercept_

comparision=pd.DataFrame({'actual':y_test,'predicted':y_pred})
print(comparision)

print("12 years salary guy is :",coeff*12+interce)



bias=regressor.score(x_train, y_train)
print(bias)

variance=regressor.score(x_test, y_test)
print(variance)



#ssr(predicted to mean )
y_mean=np.mean(y)
ssr=np.sum((y_pred-y_mean)**2)
print(ssr)


#sse(actual-predicted)
y=y[0:6]
sse=np.sum((y-y_pred)**2)
print(sse)
           
sst=ssr+sse
print(sst)
           
r_square = ssr/sst
r_square