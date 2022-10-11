"""Calculates the number of predictors of X given order p and plots.
Also plots the ratio of predictors vs data points (n). Can be used to evaluate
if overfitting is more likely, given high dimensionality"""


import matplotlib.pyplot as plt
import numpy as np



def num_pred(p):
    return (p**2 + 3*p + 2)/2


def num_pred_plot(deg,n):
    """plots the number of predictors and plots the ratio pred/n"""
    orderarray = np.zeros(deg)
    predarray = np.zeros(deg)
    ratio_pred_to_data_points = np.zeros(deg)
    for order in range(deg):
        orderarray[order] = order
        predictors = num_pred(order)
        predarray[order] = predictors
        ratio_pred_to_data_points[order] = predictors/n

        #print("order  pred")
        #print(order, pred)

    return orderarray, predarray, ratio_pred_to_data_points

deg = 30 #polynomial degree

n= 2701 #terrain data points

Plot = num_pred_plot(deg,n)
plt.plot(Plot[0], Plot[1])
plt.grid()
plt.xlabel("Poly. deg.")
plt.ylabel("beta coeff.")
plt.title("Number of beta coeff. for each poly. deg.")
plt.show()
plt.plot(Plot[0], Plot[2])
plt.xlabel("Poly. deg.")
plt.ylabel("Ratio of beta coeff. per data point")
plt.title("Ratio of beta coeff. per data point. (Terrain: 2701 points)")
plt.grid()
plt.show()
