"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.mean: float = None
        self.std: float = None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.
        """
        result = np.array(X) # X  evin genisligi ,yüksekligi
        for d in range(2, degree+1):
            result = np.column_stack((result, X**d))
        return result

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        #polynomial expansion 
        poly = self.polyfeatures(X,self.degree)

        #Data standardization 
        self.mean = poly.mean(0)
        self.std = poly.std(0)
        poly = (poly - self.mean) / self.std

        #column of 1s
        col_ones = np.ones((len(X), 1)) # 1 means the number of column 
        poly_with_X0 = np.column_stack((col_ones, poly)) 
        
        # construct reg matrix
        regMatrix = self.reg_lambda * np.eye(self.degree + 1)
        regMatrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.solve(poly_with_X0.T @ poly_with_X0 + regMatrix, poly_with_X0.T @ y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n= len(X)
        # expand X to be a n*d array with degree d
        poly = self.polyfeatures(X, self.degree)

        # compute the statistics for train and test data
        poly = (poly - self.mean) / self.std

        # add the feature row with order 0
        col_ones = np.ones((len(X), 1)) 
        poly_with_X0 = np.column_stack((col_ones, poly)) 

        # predict
        return poly_with_X0.dot(self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return ((a - b)**2).mean()


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    # Fill in errorTrain and errorTest arrays^
    for i in range (1,n):
        subXtrain = Xtrain[0:(i+1)]
        subYtrain = Ytrain[0:(i+1)]
        polyReg = PolynomialRegression(degree, reg_lambda)
        polyReg.fit(subXtrain, subYtrain)
        errorTrain[i]= mean_squared_error(polyReg.predict(subXtrain), subYtrain)
        errorTest[i]= mean_squared_error(polyReg.predict(Xtest), Ytest)
        
    return (errorTrain, errorTest)


