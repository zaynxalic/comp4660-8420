
# external library
from math import ceil
import numpy as np

"""
Functions:
lowess Fit a smooth nonparametric regression curve to a scatterplot.
Which references from 

"""
# for bigger f value, the function will be much smoother
def lowess(x, y, f, it):
    # np.seterr(divide='ignore', invalid='ignore')
    n = len(x)
    h = []
    # generate several windows in local regression to calculate
    for i in range(n):
        h.append(np.sort(np.abs(x - x[i]))[int(ceil(f * n))])
    w = np.clip(np.abs((x[:, None] - x[None, :]) /h ), 0., 1.)
    # the weight follows as W(x) = (1-|x|^3)^3
    w = (1 - np.abs(w) ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    constant = 6.0
    # add error handler
    try:
        for iteration in range(it):
            for i in range(n):
                # let the weight be delta_k * w_k
                weights = delta * w[:, i]
                b_1 = np.sum(weights * y)
                b_2 = np.sum(weights * y * x)
                b = np.array([b_1, b_2])
                
                A_1_1 = np.sum(weights)
                A_2_1 = np.sum(weights * x)
                A_1_2 = np.sum(weights * x)
                A_2_2 = np.sum(weights * x * x)
                A = np.array([
                    [A_1_1, A_2_1],
                    [A_1_2, A_2_2],
                    ])
                # if beta is singular value then, return original y value
                beta = np.linalg.solve(A, b)
                yest[i] =  beta[1] * x[i] + beta[0] 
            residuals = y - yest
            # calculate the weight: which is B(residual/6s)
            delta = np.clip(residuals / (constant * np.median(np.abs(residuals))), -1, 1)
            delta = np.square(1 - np.square(delta))
    except:
        return y
    return yest
