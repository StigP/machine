from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from seaborn import heatmap


def design_matrix(x_, y_, order):
    temp = np.arange(order + 1) + 1
    num_of_col = np.cumsum(temp)[-1]

    A = np.empty((len(x_), num_of_col))
    # 0th order
    A[:,0] = 1.0

    # 1st order
    if order >= 1:
        A[:,1] = x_
        A[:,2] = y_

    # 2nd order
    if order >= 2:
        A[:,3] = x_**2
        A[:,4] = y_**2
        A[:,5] = x_*y_

    # 3rd
    if order >= 3:
        A[:,6] = x_**3
        A[:,7] = y_**3
        A[:,8] = x_**2 * y_
        A[:,9] = x_ * y_**2

    # 4th
    if order >= 4:
        A[:,10] = x_**4
        A[:,11] = y_**4
        A[:,12] = x_**2 * y_**2
        A[:,13] = x_**3 * y_
        A[:,14] = x_ * y_**3

    # 5th
    if order >= 5:
        A[:,15] = x_**5
        A[:,16] = y_**5
        A[:,17] = x_**4 * y_
        A[:,18] = x_ * y_**4
        A[:,19] = x_**3 * y_**2
        A[:,20] = x_**2 * y_**3

    # 6th
    if order >= 6:
        A[:,21] = x_**6
        A[:,22] = y_**6
        A[:,23] = x_**5 * y_
        A[:,24] = x_ * y_**5
        A[:,25] = x_**4 * y_**2
        A[:,26] = x_**2 * y_**4
        A[:,27] = x_**3 * y_**3

    # 7th
    if order >= 7:
        A[:,28] = x_**7
        A[:,29] = y_**7
        A[:,30] = x_**6 * y_
        A[:,31] = x_ * y_**6
        A[:,32] = x_**4 * y_**3
        A[:,33] = x_**3 * y_**4
        A[:,34] = x_**5 * y_**2
        A[:,35] = x_**2 * y_**5

    # 8th
    if order >= 8:
        A[:,36] = x_**8
        A[:,37] = y_**8
        A[:,38] = x_**7 * y_
        A[:,39] = x_ * y_**7
        A[:,40] = x_**6 * y_**2
        A[:,41] = x_**2 * y_**6
        A[:,42] = x_**5 * y_**3
        A[:,43] = x_**3 * y_**5
        A[:,44] = x_**4 * y_**4

    # 9th
    if order >= 9:
        A[:,45] = x_**9
        A[:,46] = y_**9
        A[:,47] = x_**8 * y_
        A[:,48] = x_ * y_**8
        A[:,49] = x_**7 * y_**2
        A[:,50] = x_**2 * y_**7
        A[:,51] = x_**6 * y_**3
        A[:,52] = x_**3 * y_**6
        A[:,53] = x_**5 * y_**4
        A[:,54] = x_**4 * y_**5

    # 10th
    if order >= 10:
        A[:,55] = x_**10
        A[:,56] = y_**10
        A[:,57] = x_**9 * y_
        A[:,58] = x_ * y_**9
        A[:,59] = x_**8 * y_**2
        A[:,60] = x_**2 * y_**8
        A[:,61] = x_**7 * y_**3
        A[:,62] = x_**3 * y_**7
        A[:,63] = x_**6 * y_**4
        A[:,64] = x_**4 * y_**6
        A[:,65] = x_**5 * y_**5
    return A

def plot(x,y, z, z_tilde):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf_tilde = ax.plot_surface(x, y, z_tilde, cmap=cm.summer,
    linewidth=0, antialiased=False)

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.45,
    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf_tilde, shrink=0.5, aspect=5)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def OLS(order, pl):
    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)

    x, y = np.meshgrid(x,y)

    def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    np.random.seed(10)
    z = FrankeFunction(x, y) + np.random.normal(0, 0.1, x.shape)
    z_ = z.flatten().reshape(-1, 1)
    x_ = x.flatten()
    y_ = y.flatten()

    # Design matrix
    A = design_matrix(x_, y_, order)

    beta = np.linalg.pinv(A.transpose() @ A) @ A.transpose() @ z_
    z_tilde = A @ beta

    MSE_calc = np.sum((z_ - z_tilde)**2)/len(z_)
    z_mean = np.sum(z_)/len(z_)
    R2_calc = 1 - np.sum((z_ - z_tilde)**2)/np.sum((z_ - z_mean)**2)

    # if pr == True:
    #     print(f"Caclulated mean square error = {MSE_calc:.4f}")
    #     print(f"Calculated coefficient of determination = {R2_calc:.4f}")

    z_tilde = z_tilde.reshape((20,20))

    if pl == True:
        plot(x,y, z, z_tilde)

    return MSE_calc, R2_calc, z_tilde, beta

def MSE_r2_beta_plot(order):

    poly_deg = np.zeros(order, dtype='int')
    MSE_array = np.empty(order)
    r2_array = np.empty(order)
    for i in range(order):
        MSE, r2, z_tilde, beta = OLS(i+1, pl=False)
        MSE_array[i] = MSE
        r2_array[i] = r2
        poly_deg[i] = i + 1
        print(f'{np.shape(beta)}, n = {i + 1}\n')
    plt.plot(poly_deg, MSE_array, label="Train")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title(f"MSE - order of polynomia = {order}")
    plt.legend()
    plt.show()

    plt.plot(poly_deg, r2_array, label="Train")
    plt.xlabel("Polynomial degree")
    plt.ylabel("r2")
    plt.title(f"r2 score - order of polynomia = {order}")
    plt.legend()
    plt.show()

order = 10
MSE_calc, R2_calc, z_tilde, beta = OLS(order, True)
# print(beta)
MSE_r2_beta_plot(order)
