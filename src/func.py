"""Functions for project 01"""


from turtle import color
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from sklearn import linear_model
import seaborn as sns
from imageio import imread
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.utils._testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning


def data_FF(noise=True, step_size=0.05, alpha=0.05):


    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    X, Y = np.meshgrid(x, y)
    x = X.flatten().reshape(-1, 1)
    y = Y.flatten().reshape(-1, 1)
    Z = FrankeFunction(X, Y, noise, alpha, seed=3155)
    z = Z.flatten().reshape(-1, 1)
    return x, y, z

def data_terrain(row=10, col=10):
    z_mesh = np.asarray(imread('data/SRTM_data_Norway_2.tif'))
    x = np.linspace(0, z_mesh.shape[0] - 1, z_mesh.shape[0]).reshape(-1, 1)
    y = np.linspace(0, z_mesh.shape[1] - 1, z_mesh.shape[1]).reshape(-1, 1)

    x_mesh, y_mesh = np.meshgrid(x, y)

    x = x_mesh[::row, ::col].flatten().reshape(-1, 1)
    y = y_mesh[::row, ::col].flatten().reshape(-1, 1)
    z = z_mesh[::row, ::col].flatten().reshape(-1, 1)
    return x, y, z

def MSE(z, z_tilde):
    return mean_squared_error(z, z_tilde)

def r2(z, z_tilde):
    return r2_score(z, z_tilde)

def design_matrix(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def FrankeFunction(x,y, noice, alpha, seed):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noice:
        np.random.seed(seed)
        return term1 + term2 + term3 + term4 + alpha*np.random.normal(0, 1, x.shape)
    else:
        return term1 + term2 + term3 + term4


def scaling(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test

def kfold_split(arr, k):
    """
        Args
            arr (array):  The array one wants to split
            k (int):    The number of sections to split in

        Return
            A tuple of length n of arrays "equally" divided arrays
    """
    if k>len(arr):
        k=len(arr)

    n, m = divmod(len(arr), k)
    test = (arr[i*n+min(i, m):(i+1)*n+min(i+1, m)] for i in range(k))
    return tuple(test)



def OLS_Reg(x_, y_, z_, order, seed, scale, bootstrap, n_boostraps, CV=False, k=10):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_, y_, z_, test_size=0.2, random_state = seed)

    """Scaling pre use:"""
    """Scaling of z is out of convenience:"""
    z_train, z_test = scaling(z_train, z_test)

    if ((scale==True) and (CV== False)):
        x_train, x_test = scaling(x_train, x_test)
        y_train, y_test = scaling(y_train, y_test)


    if bootstrap:
        mse_train = np.empty(order)
        mse_test = np.empty(order)
        R2_train = np.empty(order)
        R2_test = np.empty(order)
        bias = np.empty(order)
        variance = np.empty(order)

        for degree in tqdm(range(order)):
            z_tilde_boot_test = np.zeros((len(z_test),n_boostraps))
            D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix
            for i in range(n_boostraps):
                x_train_boot, y_train_boot, z_train_boot = resample(x_train, y_train, z_train, random_state=i) # Bootstrap

                D_M_train_boot = design_matrix(x_train_boot, y_train_boot, degree) # Train design matrix

                beta_boot = np.linalg.pinv(D_M_train_boot.T @ D_M_train_boot) @ D_M_train_boot.T @ z_train_boot

                """QR for beta without inverting:"""

                #beta_boot = QR_beta(D_M_train_boot,z_train_boot)

                z_tilde_test = D_M_test @ beta_boot
                z_tilde_boot_test[:, i] = z_tilde_test.flatten()

            mse_test[degree] = np.mean( np.mean((z_test.reshape(-1,1) - z_tilde_boot_test)**2, axis=1, keepdims=True) )
            bias[degree] = np.mean( (z_test.reshape(-1,1) - np.mean(z_tilde_boot_test, axis=1,keepdims = True))**2)
            variance[degree] = np.mean( np.var(z_tilde_boot_test, axis=1, keepdims=True) )

            z_tilde_test = np.mean(z_tilde_boot_test, axis=1, keepdims=True).flatten()
            z_test = z_test.flatten()
            R2_test[degree] = r2(z_test, z_tilde_test)


        return mse_train, mse_test, R2_train, R2_test, bias, variance


    elif CV:
        """copy input data to new name to avoid conflicts in mupltiple runs:"""

        x_cv = x_.copy()
        y_cv = y_.copy()
        z_cv = z_.copy()

        """Cross Validation:"""
        mse_train = np.empty(order)
        mse_test = np.empty(order)
        R2_train = np.empty(order)
        R2_test = np.empty(order)
        bias = np.empty(order)
        variance = np.empty(order)


        """Pre-scaling as an option of all input data:"""
        if scale==True:
            scaler = StandardScaler()
            scaler.fit(x_cv)
            x_cv = scaler.transform(x_cv)
            scaler.fit(y_cv)
            y_cv = scaler.transform(y_cv)


        """Shuffles the data, seed must be reset for each time"""


        np.random.seed(seed)
        np.random.shuffle(x_cv)
        np.random.seed(seed)
        np.random.shuffle(y_cv)
        np.random.seed(seed)
        np.random.shuffle(z_cv)

        """k_split"""
        x_folds = kfold_split(x_cv, k)
        y_folds = kfold_split(y_cv, k)
        z_folds = kfold_split(z_cv,k)

        for degree in tqdm(range(order)):
            """Init the result at 0:"""

            mean_mse_test, mean_mse_train, mean_R2_test,mean_R2_train = 0,0,0,0
            mean_bias, mean_var = 0,0

            for fold in range (k):

                """selects the k-fold as the test set and other as test:"""
                """test fold:"""
                x_test = x_folds[fold]
                y_test = y_folds[fold]
                z_test = z_folds[fold]

                """selects the remaining folds as one training set:"""
                x_train = np.concatenate((x_folds[:fold]+x_folds[fold+1:]))
                y_train = np.concatenate((y_folds[:fold]+y_folds[fold+1:]))
                z_train = np.concatenate((z_folds[:fold]+z_folds[fold+1:]))

                """creates the design matrices:"""
                D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix
                D_M_train = design_matrix(x_train, y_train, degree) # Train design matrix

                # if scale==True:
                #     """Scaling the design matrices based on train matrix"""
                #     D_M_train, D_M_test = scaling(D_M_train, D_M_test)
                """Scaling of z is out of convenience:"""
                z_train, z_test = scaling(z_train, z_test)

                """Train and find beta and mean test MSE"""
                I = np.eye(np.shape(D_M_train)[1],np.shape(D_M_train)[1])
                beta = np.linalg.pinv(D_M_train.T @ D_M_train) @ D_M_train.T @ z_train

                """QR for beta without inverting:"""
                #beta = QR_beta(D_M_train,z_train)

                z_tilde_test = D_M_test @ beta


                z_tilde_train = D_M_train @ beta
                mean_mse_test += MSE(z_test, z_tilde_test)
                mean_R2_test += r2(z_test, z_tilde_test)
                mean_mse_train += MSE(z_tilde_train, z_train)
                mean_R2_train += r2(z_tilde_train, z_train)

                mean_bias += np.mean((z_test - np.mean(z_tilde_test, axis=0))**2)
                mean_var += np.mean(np.var(z_tilde_test, axis=0, keepdims=True))



            mse_test[degree] = mean_mse_test/k
            mse_train[degree] = mean_mse_train/k

            R2_test[degree] = mean_R2_test/k
            R2_train[degree] = mean_R2_train/k

            bias[degree] = mean_bias/k
            variance[degree] = mean_var/k

        return mse_train, mse_test, R2_train, R2_test, bias, variance

    else:
        D_M_train = design_matrix(x_train, y_train, order)
        D_M_test = design_matrix(x_test, y_test, order)

        # Train:
        beta = np.linalg.pinv(D_M_train.T @ D_M_train) @ D_M_train.T @ z_train
        z_tilde_train = D_M_train @ beta
        MSE_train = MSE(z_train, z_tilde_train)
        R2_train = r2(z_train, z_tilde_train)

        # Test:
        z_tilde_test = D_M_test @ beta
        MSE_test = MSE(z_test, z_tilde_test)
        R2_test = r2(z_test, z_tilde_test)

        bias = np.mean( (z_test - np.mean(z_tilde_test, axis=0))**2 )
        variance = np.mean(np.var(z_tilde_test, axis=0, keepdims=True))

        return MSE_train, MSE_test, R2_train, R2_test, beta, z_tilde_test, bias, variance

def Ridge_Reg(x_, y_, z_, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k=10):
    """Ridge with possible Boot or CV resampling:"""
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_, y_, z_, test_size=0.2, random_state = seed)

    """Scaling pre use:"""
    """Scaling of z is out of convenience:"""
    z_train, z_test = scaling(z_train, z_test)

    if ((scale==True) and (CV== False)):
        x_train, x_test = scaling(x_train, x_test)
        y_train, y_test = scaling(y_train, y_test)

    if bootstrap:
        lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
        MSE_degree_lambda = np.empty((order, nlambdas))
        bias = np.empty((order, nlambdas))
        variance = np.empty((order, nlambdas))

        for degree in tqdm(range(order)):
            D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix

            for j in range(nlambdas):
                z_tilde_boot_test = np.zeros((len(z_test),n_boostraps))
                lmb = lambdas[j]


                for i in range(n_boostraps):

                    x_train_boot, y_train_boot, z_train_boot = resample(x_train, y_train, z_train,random_state=i) # Bootstrap

                    D_M_train_boot = design_matrix(x_train_boot, y_train_boot, degree) # Train design matrix
                    #if scale==True:

                        #D_M_train_boot, D_M_test = scaling(D_M_train_boot, D_M_test)

                    I = np.eye(np.shape(D_M_train_boot)[1],np.shape(D_M_train_boot)[1])

                    beta_boot = np.linalg.pinv(D_M_train_boot.T @ D_M_train_boot+lmb*I) @ D_M_train_boot.T @ z_train_boot

                    z_tilde = D_M_test @ beta_boot
                    z_tilde_boot_test[:, i] = z_tilde.flatten()

                MSE_degree_lambda[degree,j] = np.mean( np.mean((z_test.reshape(-1,1) - z_tilde_boot_test)**2, axis=1, keepdims=True) )
                bias[degree,j] = np.mean( (z_test.reshape(-1,1) - np.mean(z_tilde_boot_test, axis=1,keepdims = True))**2)
                variance[degree,j] = np.mean( np.var(z_tilde_boot_test, axis=1, keepdims=True))


    elif CV:
        """Cross Validation:"""
        """copy input data to new name to avoid conflicts in mupltiple runs:"""

        x_cv = x_.copy()
        y_cv = y_.copy()
        z_cv = z_.copy()

        lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

        MSE_degree_lambda = np.empty((order, nlambdas))
        bias = np.empty((order, nlambdas))
        variance = np.empty((order, nlambdas))

        """Pre-scaling as an option of all input data:"""
        if scale==True:
            scaler = StandardScaler()
            scaler.fit(x_cv)
            x_cv = scaler.transform(x_cv)
            scaler.fit(y_cv)
            y_cv = scaler.transform(y_cv)

        """Shuffles the data, seed must be reset for each time"""
        np.random.seed(seed)
        np.random.shuffle(x_cv)
        np.random.seed(seed)
        np.random.shuffle(y_cv)
        np.random.seed(seed)
        np.random.shuffle(z_cv)

        """k_split"""
        x_folds = kfold_split(x_cv, k)
        y_folds = kfold_split(y_cv, k)
        z_folds = kfold_split(z_cv,k)

        for degree in tqdm(range(order)):
            """array in inner loop for each degree means:"""

            for j in range(nlambdas):
                lmb = lambdas[j]

                mean_MSE = 0 #init the MSE with zero.
                mean_bias = 0
                mean_variance = 0
                for fold in range (k):

                    """selects the k-fold as the test set and other as test:"""
                    """test fold:"""
                    x_test = x_folds[fold]
                    y_test = y_folds[fold]
                    z_test = z_folds[fold]
                    """selects the remaining folds as one training set:"""
                    x_train = np.concatenate((x_folds[:fold]+x_folds[fold+1:]))
                    y_train = np.concatenate((y_folds[:fold]+y_folds[fold+1:]))
                    z_train = np.concatenate((z_folds[:fold]+z_folds[fold+1:]))

                    """creates the design matrices:"""
                    D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix
                    D_M_train = design_matrix(x_train, y_train, degree) # Train design matrix

                    # if scale==True:
                    #     """Scaling the design matrices based on train matrix"""
                    #     D_M_train, D_M_test = scaling(D_M_train, D_M_test)
                    #     """Scaling of z is out of convenience:"""
                    z_train, z_test = scaling(z_train, z_test)

                    """Train and find beta and mean test MSE"""
                    I = np.eye(np.shape(D_M_train)[1],np.shape(D_M_train)[1])
                    beta = np.linalg.pinv(D_M_train.T @ D_M_train+lmb*I) @ D_M_train.T @ z_train
                    z_tilde = D_M_test @ beta

                    mean_MSE += MSE(z_test, z_tilde)
                    mean_bias += np.mean( (z_test - np.mean(z_tilde, axis=0))**2 )
                    mean_variance += np.mean( np.var(z_tilde, axis=0, keepdims=True) )

                MSE_degree_lambda[degree, j] = mean_MSE/k
                bias[degree, j] = mean_bias/k
                variance [degree, j] = mean_variance/k

    else:
        """Without Resamling:"""
        MSE_degree_lambda = np.empty((order, nlambdas))
        bias = np.empty((order, nlambdas))
        variance = np.empty((order, nlambdas))

        lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

        for degree in tqdm(range(order)):

            D_M_train = design_matrix(x_train, y_train, degree)
            D_M_test = design_matrix(x_test, y_test, degree)
            # if scale==True:
            #     """Scaling the design matrices based on train matrix"""
            #     D_M_train, D_M_test = scaling(D_M_train, D_M_test)

            for l in range(nlambdas):
                lmb = lambdas[l]

                I = np.eye(np.shape(D_M_train)[1],np.shape(D_M_train)[1])

                Ridgebeta = np.linalg.inv(D_M_train.T @ D_M_train+lmb*I) @ D_M_train.T @ z_train

                z_tilde_test = D_M_test @ Ridgebeta

                MSE_degree_lambda[degree, l] = MSE(z_test, z_tilde_test)
                bias[degree, l] = np.mean( (z_test - np.mean(z_tilde, axis=0))**2 )
                variance [degree, l] = np.mean( np.var(z_tilde, axis=0, keepdims=True) )

    return MSE_degree_lambda, bias, variance

@ignore_warnings(category=ConvergenceWarning)
def Lasso_Reg(x_, y_, z_, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k=10, LASSOiter=100, LASSOtol=0.0001):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_, y_, z_, test_size=0.2, random_state = seed)

    """Scaling pre use:"""
    """Scaling of z is out of convenience:"""
    z_train, z_test = scaling(z_train, z_test)
    if scale==True:
        # Scaling:
        x_train, x_test = scaling(x_train, x_test)
        y_train, y_test = scaling(y_train, y_test)

    if bootstrap:
        lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

        MSE_degree_lambda = np.empty((order, nlambdas))
        bias = np.empty((order, nlambdas))
        variance = np.empty((order, nlambdas))

        for degree in tqdm(range(order)):
            D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix

            for j in range(nlambdas):
                z_tilde_boot_test = np.zeros((len(z_test),n_boostraps))
                lmb = lambdas[j]

                for i in range(n_boostraps):

                    x_train_boot, y_train_boot, z_train_boot = resample(x_train, y_train, z_train,random_state=i) # Bootstrap

                    D_M_train_boot = design_matrix(x_train_boot, y_train_boot, degree) # Train design matrix
                    # if scale==True:
                    #     """Scaling the design matrices based on train matrix"""
                    #     D_M_train_boot, D_M_test = scaling(D_M_train_boot, D_M_test)

                    RegLasso = linear_model.Lasso(lmb, max_iter=LASSOiter, tol=LASSOtol)
                    RegLasso.fit(D_M_train_boot, z_train_boot)
                    z_tilde = RegLasso.predict(D_M_test)

                    z_tilde_boot_test[:, i] = z_tilde.flatten()

                MSE_degree_lambda[degree,j] = np.mean( np.mean((z_test.reshape(-1,1) - z_tilde_boot_test)**2, axis=1, keepdims=True) )
                bias[degree,j] = np.mean( (z_test.reshape(-1,1) - np.mean(z_tilde_boot_test, axis=1,keepdims = True))**2)
                variance[degree,j] = np.mean( np.var(z_tilde_boot_test, axis=1, keepdims=True))


    elif CV:
        """Cross Validation:"""
        lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

        MSE_degree_lambda = np.empty((order, nlambdas))
        bias = np.empty((order, nlambdas))
        variance = np.empty((order, nlambdas))

        """Pre-scaling as an option of all input data:"""
        if scale==True:
            scaler = StandardScaler()
            scaler.fit(x_)
            x_ = scaler.transform(x_)
            scaler.fit(y_)
            y_ = scaler.transform(y_)

        """Shuffles the data, seed must be reset for each time"""
        np.random.seed(seed)
        np.random.shuffle(x_)
        np.random.seed(seed)
        np.random.shuffle(y_)
        np.random.seed(seed)
        np.random.shuffle(z_)

        """k_split"""
        x_folds = kfold_split(x_, k)
        y_folds = kfold_split(y_, k)
        z_folds = kfold_split(z_,k)

        for degree in tqdm(range(order)):
            """array in inner loop for each degree means:"""

            for j in range(nlambdas):
                lmb = lambdas[j]

                mean_MSE = 0 #init the MSE with zero.
                mean_bias = 0
                mean_variance = 0
                for fold in range (k):

                    """selects the k-fold as the test set and other as test:"""
                    """test fold:"""
                    x_test = x_folds[fold]
                    y_test = y_folds[fold]
                    z_test = z_folds[fold]
                    """selects the remaining folds as one training set:"""
                    x_train = np.concatenate((x_folds[:fold]+x_folds[fold+1:]))
                    y_train = np.concatenate((y_folds[:fold]+y_folds[fold+1:]))
                    z_train = np.concatenate((z_folds[:fold]+z_folds[fold+1:]))

                    """creates the design matrices:"""
                    D_M_test = design_matrix(x_test, y_test, degree) # Test design matrix
                    D_M_train = design_matrix(x_train, y_train, degree) # Train design matrix

                    # if scale==True:
                    #     """Scaling the design matrices based on train matrix"""
                    #     D_M_train, D_M_test = scaling(D_M_train, D_M_test)
                    """Scaling of z is out of convenience:"""
                    z_train, z_test = scaling(z_train, z_test)

                    """Train and find beta and mean test MSE"""
                    RegLasso = linear_model.Lasso(lmb, max_iter=LASSOiter)
                    RegLasso.fit(D_M_train, z_train)
                    z_tilde = RegLasso.predict(D_M_test)

                    mean_MSE += MSE(z_test, z_tilde)
                    mean_bias += np.mean( (z_test - np.mean(z_tilde))**2 )
                    mean_variance += np.mean( np.var(z_tilde, keepdims=True) )

                MSE_degree_lambda[degree, j] = mean_MSE/k
                bias[degree, j] = mean_bias/k
                variance [degree, j] = mean_variance/k

    else:
        MSE_degree_lambda = np.empty((order, nlambdas))
        bias = np.empty((order, nlambdas))
        variance = np.empty((order, nlambdas))

        lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

        for degree in tqdm(range(order)):

            D_M_train = design_matrix(x_train, y_train, order)
            D_M_test = design_matrix(x_test, y_test, order)
            # if scale==True:
            #     """Scaling the design matrices based on train matrix"""
            #     D_M_train, D_M_test = scaling(D_M_train, D_M_test)

            for l in range(nlambdas):
                lmb = lambdas[l]

                RegLasso = linear_model.Lasso(lmb, max_iter=500000, tol= 0.01)
                RegLasso.fit(D_M_train, z_train)

                z_tilde_test = RegLasso.predict(D_M_test)

                MSE_degree_lambda[degree, l] = MSE(z_test, z_tilde_test)
                bias[degree, l] = np.mean( (z_test - np.mean(z_tilde, axis=0))**2 )
                variance [degree, l] = np.mean( np.var(z_tilde, axis=0, keepdims=True) )

    return MSE_degree_lambda, bias, variance


"""returns beta in OLS using QR factorization"""

def QR(A):
    """Returns the QR fact of A as Q and R matrices"""
    m,n = np.shape(A) #dimensions of A
    Q = np.zeros((m,n)) #init the Q matrix
    R = np.zeros((n,n)) #init the R matrix

    r_0 = np.linalg.norm(A[:,0])

    Qk = np.zeros((m,1)) #init the smallest Qk matrix
    Qk[:,0] = A[:,0]/r_0

    Rk = np.zeros((1,1)) #init the Rk matrix
    Rk[:,0] = r_0

    for k in range(n-1):

        r = np.array([Qk.T@A[:,k+1]])
        r = np.reshape(r, (k+1, 1))
        Rk = np.append(Rk,r,axis=1)

        alpha = np.linalg.norm(A[:,k+1] - (Qk@Qk.T@A[:,k+1]))

        Rrow =np.zeros((1,k+2))
        Rrow[0,k+1] = alpha
        Rk = np.append(Rk,Rrow,axis=0)

        F = np.reshape(Qk@r,-1) # reshape to vector

        q = (A[:,k+1] - (F))/alpha
        qreshaped = np.reshape(q, (m, 1))
        Qk = np.append(Qk, qreshaped, axis=1)

    R = Rk
    Q = Qk

    return Q,R


def QR_numpy(A):
    """for verification"""
    return np.linalg.qr(A)


def QR_beta(A,z):
    """computes beta using the QR algo"""
    Q,R = QR_numpy(A)
    QR_beta = np.linalg.solve(R,Q.T@z) #faster than inverting R.
    return QR_beta
