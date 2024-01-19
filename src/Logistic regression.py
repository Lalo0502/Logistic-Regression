import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import rcParams

#       ___       ___           ___       ___     
#     /\__\     /\  \         /\__\     /\  \    
#    /:/  /    /::\  \       /:/  /    /::\  \   
#   /:/  /    /:/\:\  \     /:/  /    /:/\:\  \  
#  /:/  /    /::\~\:\  \   /:/  /    /:/  \:\  \ 
# /:/__/    /:/\:\ \:\__\ /:/__/    /:/__/ \:\__\
# \:\  \    \/__\:\/:/  / \:\  \    \:\  \ /:/  /
#  \:\  \        \::/  /   \:\  \    \:\  /:/  / 
#   \:\  \       /:/  /     \:\  \    \:\/:/  /  
#    \:\__\     /:/  /       \:\__\    \::/  /   
#     \/__/     \/__/         \/__/     \/__/    

# This code is an example of logistic regression.
# A type of statistical analysis that is used to predict the probability of a binary event, such as passing or failing an exam.
# It uses the data of the students' scores on two written tests from the DMV (Department of Motor Vehicles)
# Also it trains a logistic regression model to estimate the probability of passin based on the scores.
# The code also display the graph of scores and the decision boundary of the model.



# Set the style of the plot and figure size of the plot
plt.style.use('ggplot')
rcParams['figure.figsize'] = 12, 8

# Load the data
data = pd.read_csv('DMV_Written_Tests.csv')

# Extract the scores and results from the data
scores = data[['DMV_Test_1', 'DMV_Test_2']].values
results = data['Results'].values

# Create boolean array for passed and failed students
passed = (results == 1).reshape(100, 1)
failed = (results == 0).reshape(100, 1)


def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function for logistic regression
def compute_cost(theta, x, y):
    m = len(y)
    y_pred = logistic_function(np.dot(x, theta))
    error = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))
    cost = -1 / m * sum(error)
    gradient = 1 / m * np.dot(x.transpose(), (y_pred - y))
    return cost[0], gradient

# Calculate the mean and standard deviation of scores
mean_scores = np.mean(scores, axis = 0)
std_scores = np.std(scores, axis = 0)
scores = (scores - mean_scores) / std_scores

# Get the number of rows and columns in the scores
rows = scores.shape[0]
cols = scores.shape[1]

X = np.append(np.ones((rows, 1)), scores, axis = 1)
y = results.reshape(rows, 1)

theta_init = np.zeros((cols + 1, 1))
cost, gradient = compute_cost(theta_init, X, y)

print("\nCost at initialization", cost)
print("Gradient at initialization:", gradient)

# Define the gradient descent function for logistic regression
def gradient_descent(x, y, theta, alpha, iterations):
    costs = []
    for i in range(iterations):
        cost, gradient = compute_cost(theta, x, y)
        theta -= (alpha * gradient)
        costs.append(cost)
    return theta, costs

# Perform gradient descent with an alpha of 1 and 200 iterations
theta, costs = gradient_descent(X, y, theta_init, 1, 200)

print("\n Theta after running gradient descent:", theta)
print("Resulting cost:", costs[-1])


# Create a scatter plot of the scores, with different markers for passed and failed students
# Passed students are marked with a green "^", failed students are marked with a red "X"
ax = sns.scatterplot(x = X[passed[:, 0], 1],
                     y = X[passed[:, 0], 2],
                     marker = "^",
                     color = "green",
                     s = 60)
sns.scatterplot(x = X[failed[:, 0], 1],
                     y = X[failed[:, 0], 2],
                     marker = "X",
                     color = "red",
                     s = 60)
ax.set(xlabel = "DMV Written Test 1 Scores", ylabel = "DMV Written Test 2 Scores")
ax.legend(["Passed", "Failed"])

# Calculate the x and y coordinates for the decision boundary
x_boundary = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]

sns.lineplot(x = x_boundary, y = y_boundary, color = "blue")
plt.show()

# Define a function to predict whether a student passed or failed based on their scores
def predict(theta, x):
    results = logistic_function(np.dot(x, theta))
    return results > 0.5

p = predict(theta, X)
print("Training Accuracy:", sum(p == y)[0], "%")

# Load the test scores dataset
test = np.array([50, 79])
test = (test - mean_scores) / std_scores
test = np.append(np.ones(1), test)
probability = logistic_function(np.dot(test, theta))
print("A person who scores 50 and 79 on their DMV written tests have a",
      np.round(probability[0], 2) * 100,"% probability of passing.")

test = np.array([48, 82])
test = (test - mean_scores) / std_scores
test = np.append(np.ones(1), test)
probability = logistic_function(np.dot(test, theta))
print("A person who scores 50 and 79 on their DMV written tests have a",
      np.round(probability[0], 2) * 100,"% probability of passing.")
