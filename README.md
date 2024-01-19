# About

It is my first time doing this, I would appreciate any feedback!

# Logistic Regression Example

This code demonstrates how to use logistic regression to predict the probability of a binary event, such as passing or failing an exam. It uses the data of the students' scores on two written tests from the DMV (Department of Motor Vehicles) and trains a logistic regression model to estimate the probability of passing based on the scores. The code also displays the graph of scores and the decision boundary of the model.

## Requirements

To run this code, you need to have the following libraries installed:

Try

pip install -r requirements.txt

- numpy
- matplotlib
- seaborn
- pandas

## Usage

To execute the code, simply run the following command in your terminal:

python logistic_regression.py

The code will load the data from the DMV_Written_Tests.csv file, which contains the scores and results of 100 students. The code will then perform the following steps:

- Define the logistic function, which is used to map the scores to a probability between 0 and 1.
- Define the cost function and the gradient for logistic regression, which are used to measure the error and update the parameters of the model.
- Normalize the scores by subtracting the mean and dividing by the standard deviation, to improve the convergence of the gradient descent algorithm.
- Initialize the parameters of the model (theta) to zero and perform gradient descent with a learning rate of 1 and 200 iterations, to find the optimal values of theta that minimize the cost function.
- Print the cost and the gradient at the initial and final values of theta, as well as the training accuracy of the model.
- Plot the scores and the results of the students, using different markers for passed and failed students. Also plot the decision boundary of the model, which is a straight line that separates the two classes.
- Define a function to predict whether a student passed or failed based on their scores, using the logistic function and the final values of theta.

## Output

The output of the code should look something like this:

Cost at initialization 0.6931471805599453
Gradient at initialization: [[-0.1       ]
 [-0.28122914]
 [-0.25098615]]

 Theta after running gradient descent: [[1.50850586]
 [3.5468762 ]
 [3.29383709]]
Resulting cost: 0.20349770158944375
Training Accuracy: 89 %
# Logistic-Regression
