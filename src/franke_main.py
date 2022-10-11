
"""Machine Learning on Franke data"""

"""standard imports and image reader and our functions made for this report:"""

from func import*
from plot import*



order =20
seed = 1

n_boostraps = 50
lambda_min, lambda_max = -12, 5

nlambdas = 18

noise = True
k = 10
save = True
scale = True
lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
degrees = np.arange(order)
LASSOiter = 1000
LASSOtol = 0.05

x, y, z = data_FF()
data = 'FF'

print("Number of datapoints of z:")
print (np.shape(z))

#
# """ OLS """
# scale = True #tested.
# print('OLS')
# print('MSE and R2 (No resampeling) - Starting')
#
#
# bootstrap = False
# CV = False
# mse_plot = True
# r2_plot = True
# OLS_MSE_r2_plot(x, y, z, order, seed, mse_plot, r2_plot, scale, bootstrap, n_boostraps, CV, k, save)
# print('MSE and R2 (No resampeling) - Ended')
#
# # MSE (CV)
# print('MSE (CV) - Starting')
# CV = True
# mse_plot = True
# r2_plot = False
# OLS_MSE_r2_plot(x, y, z, order, seed, mse_plot, r2_plot, scale, bootstrap, n_boostraps, CV, k, save)
# print('MSE (CV) - Ended')
#
# BVTO (BS)
# print('BVTO (BS) - Starting')
# bootstrap = True
# CV = False
# bvto_OLS(x, y, z, order, seed, scale, bootstrap, n_boostraps, save)
# print('BVTO (BS) - Ended')
#
# Beta (No resampeling)
# print('Beta (No resampeling) - Strating')
# beta_plot(x, y, z, order, seed, save, scale)
# print('Beta (No resampeling) - Ended\n')
#
#
# """ RIDGE """
# scale = True
# print('RIDGE')
# # BVTO (BS)
# print('BVTO (BS) - Strated')
# bootstrap = True
# CV = False
# method = 'RIDGE'
# bvto_RIDGE_LASSO(x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k, method, LASSOiter, LASSOtol, save)
# print('BVTO (BS) - Ended')
#
# # Heatmap around the best polynomial dregree and best lambda (BS)
# print('Heatmap around the best polynomial dregree and best lambda (BS) - Started')
# MSE_heatmap_RIDGE_LASSO(method, x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,  CV, k, LASSOiter, LASSOtol, save, data)
# print('Heatmap around the best polynomial dregree and best lambda (BS) - Ended')
#
# # Heatmap around the best polynomial dregree and best lambda (CV)
# print('# Heatmap around the best polynomial dregree and best lambda (CV) - Started')
# bootstrap = False
# CV = True
# MSE_heatmap_RIDGE_LASSO(method, x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,  CV, k, LASSOiter, LASSOtol, save, data)
# print('# Heatmap around the best polynomial dregree and best lambda (CV) - Ended\n')


""" LASSO CV """
# print('LASSO')
# method = 'LASSO'
# scale = True
# # Heatmap around the best polynomial dregree and best lambda (CV)
# bootstrap = False
# CV = True
# # print('# Heatmap around the best polynomial dregree and best lambda (CV) - Started')
# MSE_heatmap_RIDGE_LASSO(method, x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,  CV, k, LASSOiter, LASSOtol,save, data)
# print('# Heatmap around the best polynomial dregree and best lambda (CV) - Ended')

#
""" LASSO BOOT, can not be run at the same time as LASSO CV above, values are somehow copied in to the equation."""

scale = True
#BVTO (BS)
print('BVTO (BS) - Strated')
bootstrap = True
CV = False
method = 'LASSO'
bvto_RIDGE_LASSO(x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,CV, k, method, LASSOiter, LASSOtol, save)
print('BVTO (BS) - Ended')
#
#
# # Heatmap around the best polynomial dregree and best lambda (BS)
# print('Heatmap around the best polynomial dregree and best lambda (BS) - Started')
# MSE_heatmap_RIDGE_LASSO(method, x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,  CV, k, LASSOiter, LASSOtol,save, data)
# print('Heatmap around the best polynomial dregree and best lambda (BS) - Ended')
