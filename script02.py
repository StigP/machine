from turtle import color
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

#plt.style.use('seaborn')
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def MSE_R2(poly_degrees, MSE_train, MSE_test, R2_train, R_test, title, fname):
    step = 3
    poly_degrees_new = np.arange(poly_degrees[0], len(poly_degrees), step)
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.suptitle(title, fontsize=20)
    ax.set_title('Mean Square Error', fontsize=18)
    ax.plot(poly_degrees, MSE_train, label="Training data")
    ax.plot(poly_degrees, MSE_test, label="Test data")
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.legend(fontsize=18)

    ax = axes[1]
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title('R2 score', fontsize=18)
    ax.plot(poly_degrees, R2_train, label="Training data")
    ax.plot(poly_degrees, R_test, label="Test data")
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('R2', fontsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

def FrankPlot(Z, x, y):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

def MSE(z, z_tilde):
    return np.sum((z - z_tilde)**2)/len(z)

def r2(z, z_tilde):
    z_mean = np.sum(z)/len(z)
    return 1 - np.sum((z - z_tilde)**2)/np.sum((z - z_mean)**2) #STIG: if zero?

def design_matrix(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2) # Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def plot_OLS(x,y, z, z_tilde):
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

def print_MSE_comparison(MSE_pre_scaling, MSE_post_scaling):
    print('Mean square errer:')
    print(f'Before scaling = {MSE_pre_scaling:.5f}')
    print(f'After scaling  = {MSE_post_scaling:.5f}\n')

def print_r2_comparison(r2_pre_scaling, r2_post_scaling):
    print('R2 score:')
    print(f'Before scaling = {r2_pre_scaling:.5f}')
    print(f'After scaling  = {r2_post_scaling:.5f}\n')

def scaling(D_M_train, D_M_test):
    scaler = StandardScaler()

    scaler.fit(D_M_train[:, 1:])

    D_M_train[:, 1:] = scaler.transform(D_M_train[:, 1:])
    D_M_test[:, 1:] = scaler.transform(D_M_test[:, 1:])

    return D_M_train, D_M_test

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def noise(alpha, x):
    """
    Args:
        alpha   (float):    Scaling factor for scaling the N(0,1) noise
        x       (array):    Used to get correct shape

    Return:
        Scaled random samples from a normal (Gaussian) distribution.
    """
    return alpha*np.random.normal(0, 1, x.shape)

def OLS(x, y, z, order, seed, scale):
    # Assigning z as the Franke function and flattening z, x and y
    """related to problem b)"""
    z_ = z.flatten().reshape(-1, 1)
    x_ = x.flatten()
    y_ = y.flatten()

    # Create the design matrix
    D_M = design_matrix(x_, y_, order)

    # Split into train and test samples
    D_M_train, D_M_test, z_train, z_test = train_test_split(D_M, z_, test_size=0.2)

    if scale==True:
        # Scaling:
        D_M_train, D_M_test = scaling(D_M_train, D_M_test)

    # Train:
    beta = np.linalg.pinv(D_M_train.T @ D_M_train) @ D_M_train.T @ z_train
    z_tilde_train = D_M_train @ beta
    MSE_train = MSE(z_train, z_tilde_train)
    R2_train = r2(z_train, z_tilde_train)

    # Test:
    z_tilde_test = D_M_test @ beta
    MSE_test = MSE(z_test, z_tilde_test)
    R2_test = r2(z_test, z_tilde_test)


    return MSE_train, MSE_test, R2_train, R2_test, beta

def MSE_r2_beta_plot(order, M, r, b, n):
    """
        This function takes the order og polynomail degree and plots the MSE,
        R2 score and beta valuse for the OLS against the polynomial degree.

    Args:
        order   (int):  The highest polynomial order
        M       (bool): True or False for plotting the MSE
        r       (bool): True or False for plotting the R2 score
        b       (bool): True or False for plotting beta0, beta1,
                        and beta2 against the polynomial degrees
    Return:
        None
    """
    # Not scaled
    MSE_train_array = np.empty(order)
    MSE_test_array = np.empty(order)

    R2_train_array = np.empty(order)
    R2_test_array = np.empty(order)

    B0 = np.empty(order)
    B1 = np.empty(order)
    B2 = np.empty(order)

    # Scaled
    MSE_train_array_scaled = np.empty(order)
    MSE_test_array_scaled = np.empty(order)

    R2_train_array_scaled = np.empty(order)
    R2_test_array_scaled = np.empty(order)

    B0_scaled = np.empty(order)
    B1_scaled = np.empty(order)
    B2_scaled = np.empty(order)

    # Polynomial degree
    poly_deg = np.arange(order) + 1

    MSE_train_mean = np.zeros(order)
    MSE_test_mean = np.zeros(order)

    MSE_train_scaled_mean = np.zeros(order)
    MSE_test_scaled_mean = np.zeros(order)

    seeds = np.arange(n) + 1

    for i in tqdm(range(order)):
        order = i+1
        for j in range(n):
            seed = seeds[j]

            MSE_train_scaled, MSE_test_scaled, R2_train_scaled, R2_test_scaled, beta_scaled = OLS(x, y, z, order, seed, scale=True)
            MSE_train, MSE_test, R2_train, R2_test, beta = OLS(x, y, z, order, seed, scale=False)

            MSE_train_mean[i] += MSE_train
            MSE_test_mean[i] += MSE_test

            MSE_train_scaled_mean[i] += MSE_train_scaled
            MSE_test_scaled_mean[i] += MSE_test_scaled


        MSE_train_scaled, MSE_test_scaled, R2_train_scaled, R2_test_scaled, beta_scaled = OLS(x, y, z, order, seed, scale=True)
        MSE_train, MSE_test, R2_train, R2_test, beta = OLS(x, y, z, order, seed, scale=False)

        # Not scaled
        MSE_train_array[i] = MSE_train
        MSE_test_array[i] = MSE_test

        R2_train_array[i] = R2_train
        R2_test_array[i] = R2_test

        B0[i] = beta[0]
        B1[i] = beta[1]
        B2[i] = beta[2]

        # Scaled
        MSE_train_array_scaled[i] = MSE_train_scaled
        MSE_test_array_scaled[i] = MSE_test_scaled

        R2_train_array_scaled[i] = R2_train_scaled
        R2_test_array_scaled[i] = R2_test_scaled

        B0_scaled[i] = beta_scaled[0]
        B1_scaled[i] = beta_scaled[1]
        B2_scaled[i] = beta_scaled[2]

        # MSE_R2(poly_deg, MSE_train_array, MSE_test_array, R2_train_array, R2_test_array, 'Something', fname=f'plots/OLS_MSE_R2score{i+1}.pdf')

    MSE_train_mean = MSE_train_mean/n
    MSE_test_mean = MSE_test_mean/n

    MSE_train_scaled_mean = MSE_train_scaled_mean/n
    MSE_test_scaled_mean = MSE_test_scaled_mean/n

    if M==True:
        plt.plot(poly_deg, MSE_train_mean, color='darkorange', label="Train mean")
        plt.plot(poly_deg, MSE_test_mean, color='dodgerblue', label="Test mean")
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.title("Mean Square Error - not scaled (mean)")
        plt.legend()
        plt.show()

        plt.plot(poly_deg, MSE_train_scaled_mean, color='darkorange', label="Train scaled mean")
        plt.plot(poly_deg, MSE_test_scaled_mean, color='dodgerblue', label="Test scaled mean")
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.title("Mean Square Error - scaled (mean)")
        plt.legend()
        plt.show()

        # plt.plot(poly_deg, MSE_train_array, color='darkorange', label="Train")
        # plt.plot(poly_deg, MSE_test_array, color='dodgerblue', label="Test")
        # plt.xlabel("Polynomial degree")
        # plt.ylabel("MSE")
        # plt.title("Mean Square Error - not scaled")
        # plt.legend()
        # plt.show()

        # plt.plot(poly_deg, MSE_train_array_scaled, color='darkorange', label="Train scaled")
        # plt.plot(poly_deg, MSE_test_array_scaled, color='dodgerblue', label="Test scaled")
        # plt.xlabel("Polynomial degree")
        # plt.ylabel("MSE")
        # plt.title("Mean Square Error - scaled")
        # plt.legend()
        # plt.show()

    if r==True:
        plt.plot(poly_deg, R2_train_array, color='darkorange', label="Train")
        plt.plot(poly_deg, R2_test_array, color='dodgerblue', label="Test")
        plt.xlabel("Polynomial degree")
        plt.ylabel(r"$R^2$")
        plt.title(r"$R^2$ score - not scaled")
        plt.legend()
        plt.show()

        plt.plot(poly_deg, R2_train_array_scaled, color='darkorange', label="Train scaled")
        plt.plot(poly_deg, R2_test_array_scaled, color='dodgerblue', label="Test scaled")
        plt.xlabel("Polynomial degree")
        plt.ylabel(r"$R^2$")
        plt.title(r"$R^2$ score - scaled")
        plt.legend()
        plt.show()

    if b==True:
        plt.plot(poly_deg, B0, color = 'dodgerblue', label="Beta 0")
        plt.plot(poly_deg, B1, color = 'forestgreen', label="Beta 1")
        plt.plot(poly_deg, B2, color = 'orangered', label="Beta 2")

        plt.xlabel("Polynomial degree")
        plt.ylabel("beta")
        plt.title(f"Beta values against order of polynomia")
        plt.legend()
        plt.show()

        plt.plot(poly_deg, B0_scaled, color = 'dodgerblue', label="Beta 0 scaled")
        plt.plot(poly_deg, B1_scaled, color = 'forestgreen', label="Beta 1 scaled")
        plt.plot(poly_deg, B2_scaled, color = 'orangered', label="Beta 2 scaled")

        plt.xlabel("Polynomial degree")
        plt.ylabel("beta")
        plt.title(f"Beta values against order of polynomia")
        plt.legend()
        plt.show()

def bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, order, n_boostraps):

    MSE = np.zeros(order)
    bias = np.zeros(order)
    variance = np.zeros(order)
    polydegree = np.arange(order)


    z_tilde_boot = np.empty((len(z_test),n_boostraps))

    for degree in tqdm(range(order)):
        D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix
        for i in range(n_boostraps):
            x_train_boot, y_train_boot, z_train_boot = resample(x_train, y_train, z_train) # Bootstrap

            D_M_train_boot = design_matrix(x_train_boot, y_train_boot, degree) # Train design matrix

            beta_boot = np.linalg.pinv(D_M_train_boot.T @ D_M_train_boot) @ D_M_train_boot.T @ z_train_boot

            z_tilde = D_M_test @ beta_boot
            z_tilde_boot[:, i] = z_tilde.flatten()

        MSE[degree] = np.mean( np.mean((z_test - z_tilde_boot)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (z_test - np.mean(z_tilde_boot, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_tilde_boot, axis=1, keepdims=True) )

    plt.plot(polydegree, MSE, label='MSE')
    plt.plot(polydegree, bias, label='bias')
    plt.plot(polydegree, variance, label='variance')
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE, bias and variance")
    plt.title("Bias-variance trade-of")
    plt.legend()
    plt.show()



def kfold_split(a, n):
    """
        Args
            a (array):  The array one wants to split
            n (int):    The number of sections to split in

        Return
            A tuple of length n of arrays "equally" divided arrays
    """
    if n>len(a): #STIG
        n=len(a) #STIG
    k, m = divmod(len(a), n)
    test = (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    return tuple(test)


def cross_validation(x,y,z, order,k):
    """splits data in k-folds and sets the kth fold as testing
    and the others as train. Does k interations and returns the means.
    Scale is not yet implemented."""

    """verifies k:"""
    if k<=1: #STIG
        raise ValueError("k must be greater than 1")
    if type(k)== float:
        raise ValueError("k must be an integer")

    """shuffle DISCUSS STIG"""
    np.random.seed(seed) #sets the seed for shuffle STIG
    np.random.shuffle(x)
    np.random.shuffle(y)
    np.random.shuffle(z)

    """k_split"""
    x_folds = kfold_split(x, k)
    y_folds = kfold_split(y, k)
    z_folds = kfold_split(z,k)

    """array for MSE on train and test:"""
    MSE_train = np.zeros(order)
    MSE_test = np.zeros(order)
    polydegree = np.arange(order)

    """bias and variance not added in the output her. Does 1d) ask for it?"""
    #bias = np.zeros(order)
    #variance = np.zeros(order)

    for degree in range(order):
        """array in inner loop for each degree means:"""
        MSE_train_deg = np.zeros(k)
        MSE_test_deg = np.zeros(k)

        for fold in range (k):
            """selects the k-fold as the test set and other as test:"""
            """test fold:"""
            D_M_test = design_matrix(x_folds[fold], y_folds[fold], degree) # Test design matrix
            z_test = z_folds[fold]
            """selects the remaining folds as training set:"""
            x_train = np.delete(x_folds,fold, axis=0)
            y_train = np.delete(y_folds,fold, axis=0)
            z_train = np.delete(z_folds,fold, axis=0)
            """there is probably a shorter code for this?"""
            x_train = np.concatenate(x_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
            z_train = np.concatenate(z_train, axis=0)

            D_M_train = design_matrix(x_train, y_train, degree) # Train design matrix

            """Scaling to be added but ran in to issues here"""
            # if scale==True:
            #Scaling:
            #D_M_train, D_M_test = scaling(D_M_train, D_M_test)

            """Train"""
            beta = np.linalg.pinv(D_M_train.T @ D_M_train) @ D_M_train.T @ z_train
            z_tilde_train = D_M_train @ beta

            #Train MSE:
            MSE_train_deg[fold] = np.sum((z_train - z_tilde_train)**2)/len(z_train)
            #R2_train = r2(z_train, z_tilde_train)

            """Test"""
            z_tilde_test = D_M_test @ beta
            MSE_test_deg[fold] = np.sum((z_test - z_tilde_test)**2)/len(z_test)
            #R2_test = r2(z_test, z_tilde_test)
        """MSE mean of the training sets"""
        MSE_train_deg_ = np.mean(MSE_train_deg)

        """MSE mean of the test sets"""
        MSE_test_deg_ = np.mean(MSE_test_deg)
        """MSE for each degree"""
        MSE_train[degree] = MSE_train_deg_
        MSE_test[degree] = MSE_test_deg_

    plt.plot(polydegree, MSE_train, label='MSE_train')
    plt.plot(polydegree, MSE_test, label='MSE_test')
    plt.xlabel("Polynomial degree")
    plt.ylabel(f"MSE with a {k}-fold cross validation")
    plt.legend()
    plt.show()

    print(MSE_train, MSE_test)




if __name__=="__main__":
    order = 14
    n = 500
    n_boostraps = 100
    seed = 12345

    step_size = 0.075


    # Makeing data
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)
    x, y = np.meshgrid(x,y)


    np.random.seed(seed) #sets the seed STIG
    z = FrankeFunction(x, y) + noise(0.01, x)
    z_ = z.flatten().reshape(-1, 1)
    x_ = x.flatten()
    y_ = y.flatten()


    """cross val:"""
    k = 5 #k-folds
    CROSSVAL = cross_validation(x_, y_, z_, order, k)




    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_, y_, z_, test_size=0.2)

    # Preforms a bootstrap and plots the MSE, bias and variance



    bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, order, n_boostraps)

    # Plots and avrage of the MSE and R2
    MSE_r2_beta_plot(order, M=True, r=True, b=True, n=n)

    # Print out the MSE and R2 score for the train and test
    MSE_train_scaled, MSE_test_scaled, R2_train_scaled, R2_test_scaled, beta_scaled = OLS(x, y, z, order, seed, scale=True)
    MSE_train, MSE_test, R2_train, R2_test, beta = OLS(x, y, z, order, seed, scale=False)
    print(f"Train:")
    print_MSE_comparison(MSE_train, MSE_train_scaled)
    print_r2_comparison(R2_train, R2_train_scaled)
    print("\n")
    print(f"Test:")
    print_MSE_comparison(MSE_test, MSE_test_scaled)
    print_r2_comparison(R2_test, R2_test_scaled)


    # Plot the Frank function
    Z = FrankeFunction(x, y)
    FrankPlot(Z, x, y)
    plt.show()
