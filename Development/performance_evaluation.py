import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat,savemat
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import torch
import matplotlib.pyplot as plt
import statsmodels.api as sm

Dir_processed_data = 'H:\\My Drive\\Paper\\MG Denoising\\Data\\Full Set'
Dir_indices = 'H:\\My Drive\\Paper\\MG Denoising\\'
Dir_modeling_results = 'H:\\My Drive\\Paper\\MG Denoising\\Selected Prediction 2'

file_vel_x = 'ConvAutoencoder_fullset_vel_x_channels_16_32_64_kernel_size_10_initial lr_0.005_epoch_500batchnorm_l2_1e-2.mat'
file_vel_y = 'ConvAutoencoder_fullset_symmetry_vel_y_channels_16_32_64_kernel_size_10_initial lr_0.005_epoch_500batchnorm_l2_1e-2.mat'
file_vel_z = 'ConvAutoencoder_noiselinear_fullset_AUG_vel_z_channels_20_40_80_kernel_size_10_initial lr_0.01_epoch_700batchnorm_l2_1e-2.mat'
file_lin_x = 'ConvAutoencoder_fullset_lin_x_channels_16_32_64_kernel_size_10_initial lr_0.005_epoch_500batchnorm_l2_1e-2.mat'
file_lin_y = 'ConvAutoencoder_noiselinear_fullset_AUG_lin_y_channels_20_40_80_kernel_size_10_initial lr_0.005_epoch_600batchnorm_l2_1e-2.mat'
file_lin_z = 'ConvAutoencoder_fullset_lin_z_channels_32_64_128_kernel_size_10_initial lr_0.005_epoch_300batchnorm_l2_1e-2.mat'

def extract_independent_impacts(filename):
    results = loadmat(filename)
    independent_impacts_id = np.linspace(start=0,stop=490,num=50).astype(int)
    raw_MG = results['X_test'].squeeze()[independent_impacts_id,:]
    denoised_MG = results['Y_test_pred'][independent_impacts_id,:]
    ATD = results['Y_test'][independent_impacts_id,:]

    assert raw_MG.shape[1]==100
    assert denoised_MG.shape[1]==100
    assert ATD.shape[1]==100
    return raw_MG,denoised_MG,ATD

def compute_SNR_out(y_true, y_pred):
    SNR_out = 10*np.log10(np.sum(np.square(y_true),axis=1)/np.sum(np.square(y_true-y_pred),axis=1)) #(N,)
    return np.mean(SNR_out)

def compute_SNR_in(y_true, raw):
    SNR_in = 10*np.log10(np.sum(np.square(y_true),axis=1)/np.sum(np.square(y_true-raw),axis=1)) #(N,)
    return np.mean(SNR_in)

def evaluate_pointwise(y_true,y_pred):
    MAE = mean_absolute_error(y_true,y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
    R2 = r2_score(y_true,y_pred)
    return MAE, RMSE, R2

def evaluate_peak(y_true, y_pred):
    MAE = mean_absolute_error(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1))
    RMSE = np.sqrt(mean_squared_error(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1)))
    R2 = r2_score(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1))
    return MAE, RMSE, R2

def compute_magnitude(x,y,z):
    return np.sqrt(np.square(x)+np.square(y)+np.square(z))

if __name__=='__main__':
    # #1. Individual axis
    # os.chdir(Dir_modeling_results)
    # raw_MG, denoised_MG, ATD = extract_independent_impacts(file_lin_y)
    # mean_MAE, mean_RMSE, mean_R2 = evaluate_pointwise(ATD, denoised_MG)
    # mean_MAE_baseline, mean_RMSE_baseline, mean_R2_baseline = evaluate_pointwise(ATD, raw_MG)
    # MAE, RMSE, R2 = evaluate_peak(ATD,denoised_MG)
    # MAE_baseline, RMSE_baseline, R2_baseline = evaluate_peak(ATD,raw_MG)
    # SNR_in = compute_SNR_in(ATD,raw_MG)
    # SNR_out = compute_SNR_in(ATD, denoised_MG)
    #
    # print(';'.join([str(round(mean_MAE,3)), str(round(mean_RMSE,3)), str(round(mean_R2,3)), str(round(MAE,3)),str(round(RMSE,3)),str(round(R2,3)),str(round(SNR_out,3))]))
    # print(';'.join([str(round(mean_MAE_baseline,3)), str(round(mean_RMSE_baseline,3)), str(round(mean_R2_baseline,3)), str(round(MAE_baseline,3)),str(round(RMSE_baseline,3)),str(round(R2_baseline,3)),str(round(SNR_in,3))]))
    #
    # savemat('Linear Acceleration Y.mat',{'raw_MG':raw_MG,'ATD': ATD,'denoised_MG':denoised_MG})

    # # 2. Magnitude
    # os.chdir(Dir_modeling_results)
    # raw_MG_x, denoised_MG_x, ATD_x = extract_independent_impacts(file_lin_x)
    # raw_MG_y, denoised_MG_y, ATD_y = extract_independent_impacts(file_lin_y)
    # raw_MG_z, denoised_MG_z, ATD_z = extract_independent_impacts(file_lin_z)
    # raw_MG = compute_magnitude(raw_MG_x,raw_MG_y,raw_MG_z)
    # denoised_MG = compute_magnitude(denoised_MG_x,denoised_MG_y,denoised_MG_z)
    # ATD = compute_magnitude(ATD_x, ATD_y, ATD_z)
    #
    # mean_MAE, mean_RMSE, mean_R2 = evaluate_pointwise(ATD, denoised_MG)
    # mean_MAE_baseline, mean_RMSE_baseline, mean_R2_baseline = evaluate_pointwise(ATD, raw_MG)
    # MAE, RMSE, R2 = evaluate_peak(ATD, denoised_MG)
    # MAE_baseline, RMSE_baseline, R2_baseline = evaluate_peak(ATD, raw_MG)
    # SNR_in = compute_SNR_in(ATD, raw_MG)
    # SNR_out = compute_SNR_in(ATD, denoised_MG)
    #
    # print(';'.join([str(round(mean_MAE, 3)), str(round(mean_RMSE, 3)), str(round(mean_R2, 3)), str(round(MAE, 3)),
    #                 str(round(RMSE, 3)), str(round(R2, 3)), str(round(SNR_out, 3))]))
    # print(';'.join(
    #     [str(round(mean_MAE_baseline, 3)), str(round(mean_RMSE_baseline, 3)), str(round(mean_R2_baseline, 3)),
    #      str(round(MAE_baseline, 3)), str(round(RMSE_baseline, 3)), str(round(R2_baseline, 3)), str(round(SNR_in, 3))]))
    #
    # savemat('Linear Acceleration Magnitude.mat', {'raw_MG': raw_MG, 'ATD': ATD, 'denoised_MG': denoised_MG})

    #Draw Bland Altman Plot for Kinematics
    os.chdir(Dir_modeling_results)
    results = loadmat('Linear Acceleration Magnitude')
    raw_MG, denoised_MG, ATD = results['raw_MG'],results['denoised_MG'],results['ATD']

    plt.figure(figsize=(9, 6))
    sm.graphics.mean_diff_plot(raw_MG.reshape(-1,), ATD.reshape(-1,), scatter_kwds={'s': 0.4, 'alpha': 0.8}, mean_line_kwds={})
    plt.xlim([0, 40])
    plt.ylim([-30, 30])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Mean',fontsize=22)
    plt.ylabel('Difference',fontsize=22)
    plt.savefig('Lin Mag Pointwise Bland Altman.png', bbox_inches='tight', dpi=1200)
    plt.savefig('Lin Mag Pointwise Bland Altman.pdf', bbox_inches='tight', dpi=1200)
    plt.show()

    plt.figure(figsize=(9, 6))
    sm.graphics.mean_diff_plot(denoised_MG.reshape(-1, ), ATD.reshape(-1, ), scatter_kwds={'s': 0.4, 'alpha': 0.8})
    plt.xlim([0, 40])
    plt.ylim([-30, 30])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Mean', fontsize=22)
    plt.ylabel('Difference', fontsize=22)
    plt.savefig('Denoised Lin Mag Pointwise Bland Altman.png', bbox_inches='tight', dpi=1200)
    plt.savefig('Denoised Lin Mag Pointwise Bland Altman.pdf', bbox_inches='tight', dpi=1200)
    plt.show()