import numpy as np
from scipy.io import loadmat, savemat
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

Dir_data_1 = 'H:\\Shared drives\\Impact Testing 2\\MG Comparison\\Opro'
Dir_data_2 = 'H:\\Shared drives\\Impact Testing 2\\MG Comparison\\Gameon'
Dir_data_3 = 'H:\\My Drive\\Paper\\MG Denoising\\Data\\Additional Test'
Dir_processed_data = 'H:\\My Drive\\Paper\\MG Denoising\\Data'
Dir_processed_data_addtest = 'H:\\My Drive\\Paper\\MG Denoising\\Data\\Additional Test'

window_width = 100
stride = 5

def augment_data(folder: str, filename: str, width: int, stride: int):
    '''
    Partition the entire period of recordings into multiple data samples with sliding windows
    :param filename: The input raw .mat file which includes the mouthguard/test dummy sensor measurements
    :param width: The width of the sliding windows
    :param stride: The stride of the sliding windows for data augmentation
    :return: save the processed data in the .npy form
    '''
    os.chdir(folder)
    data = loadmat(filename)
    if filename.startswith('MG'):
        lin = data['mg_data']['lin_acc_CG'][0]
        vel = data['mg_data']['ang_vel'][0]
        acc = data['mg_data']['ang_acc'][0]
    else:
        lin = data['impact']['lin_acc_CG'][0]
        vel = data['impact']['ang_vel'][0]
        acc = data['impact']['ang_acc'][0]

    lin_x = []
    lin_y = []
    lin_z = []
    vel_x = []
    vel_y = []
    vel_z = []
    acc_x = []
    acc_y = []
    acc_z = []

    #Extract frames
    for impact in range(lin.shape[0]):
        for start_idx in range(0,200-width,stride):

            lin_x.append(lin[impact][start_idx:start_idx + width, 0].reshape(1,-1))
            lin_y.append(lin[impact][start_idx:start_idx + width, 1].reshape(1,-1))
            lin_z.append(lin[impact][start_idx:start_idx + width, 2].reshape(1,-1))

            vel_x.append(vel[impact][start_idx:start_idx + width, 0].reshape(1,-1))
            vel_y.append(vel[impact][start_idx:start_idx + width, 1].reshape(1,-1))
            vel_z.append(vel[impact][start_idx:start_idx + width, 2].reshape(1,-1))

            acc_x.append(acc[impact][start_idx:start_idx + width, 0].reshape(1,-1))
            acc_y.append(acc[impact][start_idx:start_idx + width, 1].reshape(1,-1))
            acc_z.append(acc[impact][start_idx:start_idx + width, 2].reshape(1,-1))

    #Concatenate the list of frames
    lin_x = np.concatenate(lin_x,axis=0)
    lin_y = np.concatenate(lin_y,axis=0)
    lin_z = np.concatenate(lin_z,axis=0)

    vel_x = np.concatenate(vel_x,axis=0)
    vel_y = np.concatenate(vel_y,axis=0)
    vel_z = np.concatenate(vel_z,axis=0)

    acc_x = np.concatenate(acc_x,axis=0)
    acc_y = np.concatenate(acc_y,axis=0)
    acc_z = np.concatenate(acc_z,axis=0)

    #Store the results
    os.chdir(Dir_processed_data)
    savedir = filename[:-4] + '_frames.mat'
    savedict = {'lin_x':lin_x,'lin_y': lin_y, 'lin_z': lin_z,
                'vel_x':vel_x,'vel_y': vel_y, 'vel_z': vel_z,
                'acc_x':acc_x,'acc_y': acc_y, 'acc_z': acc_z}
    savemat(file_name=savedir,mdict=savedict)

    os.chdir(folder)

def process_raw_signals(folders: list):
    '''
    This function partitions the raw signals into multiple frames with overlaps in a sliding window manner
    :param folders: a list of directory (folders that contain the raw signals)
    :return: None
    '''

    for folder in folders:
        os.chdir(folder)
        for root, dirs, files in os.walk(".", topdown=False):
            for name in tqdm(files):
                augment_data(folder=folder, filename=name, width=window_width, stride=stride)
    return None

def report_file_name(folders: list):
    '''
    This function partitions the raw signals into multiple frames with overlaps in a sliding window manner
    :param folders: a list of directory (folders that contain the raw signals)
    :return: None
    '''

    for folder in folders:
        os.chdir(folder)
        for root, dirs, files in os.walk(".", topdown=False):
            for name in tqdm(files):
                data = loadmat(name)
                if name.startswith('MG'):
                    lin = data['mg_data']['lin_acc_CG'][0]
                else:
                    lin = data['impact']['lin_acc_CG'][0]
                print('Shape of file: ',name,' ', lin.shape[0])
    return None

def extract_samples(folder: str, filename: str):
    '''
    Extract the framed data from different .mat files and return the linear acceleration at head center of gravity, angular velocity and angular acceleration.
    :param folder: Folder directory of the .mat files
    :param filename: File name of specific .mat file
    :return: lin_x, lin_y, lin_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z
    '''
    curr_dir = os.getcwd()
    os.chdir(folder)
    data = loadmat(filename)
    os.chdir(curr_dir)
    lin_x = data['lin_x']
    lin_y = data['lin_y']
    lin_z = data['lin_z']
    vel_x = data['vel_x']
    vel_y = data['vel_y']
    vel_z = data['vel_z']
    acc_x = data['acc_x']
    acc_y = data['acc_y']
    acc_z = data['acc_z']
    return lin_x, lin_y, lin_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z

def construct_entire_set(folder: str):
    '''
    The folder extract all the samples and construct the entire dataset of X and Y
    :param folder: The folder with all the samples in it
    :return: X and Y
    '''
    os.chdir(folder)

    X_lin_x = []
    X_lin_y = []
    X_lin_z = []
    X_vel_x = []
    X_vel_y = []
    X_vel_z = []
    X_acc_x = []
    X_acc_y = []
    X_acc_z = []

    Y_lin_x = []
    Y_lin_y = []
    Y_lin_z = []
    Y_vel_x = []
    Y_vel_y = []
    Y_vel_z = []
    Y_acc_x = []
    Y_acc_y = []
    Y_acc_z = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in tqdm(files):
            #Extract the samples for the X dataset
            if name.startswith('MG') and "frames" in name:
                X_file = name
                print('Processing file: ',name)
                lin_x, lin_y, lin_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = extract_samples(folder=Dir_processed_data_addtest,
                                                                                                filename=X_file)
                X_lin_x.append(lin_x)
                X_lin_y.append(lin_y)
                X_lin_z.append(lin_z)
                X_vel_x.append(vel_x)
                X_vel_y.append(vel_y)
                X_vel_z.append(vel_z)
                X_acc_x.append(acc_x)
                X_acc_y.append(acc_y)
                X_acc_z.append(acc_z)

                #Extrac the samples for the Y dataset
                Y_file = 'DTS' + name[2:]
                lin_x, lin_y, lin_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = extract_samples(folder=Dir_processed_data_addtest,
                                                                                                filename=Y_file)
                Y_lin_x.append(lin_x)
                Y_lin_y.append(lin_y)
                Y_lin_z.append(lin_z)
                Y_vel_x.append(vel_x)
                Y_vel_y.append(vel_y)
                Y_vel_z.append(vel_z)
                Y_acc_x.append(acc_x)
                Y_acc_y.append(acc_y)
                Y_acc_z.append(acc_z)

    np.save('X_lin_2_x.npy',np.concatenate(X_lin_x,axis=0))
    np.save('X_lin_2_y.npy',np.concatenate(X_lin_y,axis=0))
    np.save('X_lin_2_z.npy', np.concatenate(X_lin_z,axis=0))
    np.save('X_vel_2_x.npy', np.concatenate(X_vel_x, axis=0))
    np.save('X_vel_2_y.npy', np.concatenate(X_vel_y, axis=0))
    np.save('X_vel_2_z.npy', np.concatenate(X_vel_z, axis=0))
    np.save('X_acc_2_x.npy', np.concatenate(X_acc_x, axis=0))
    np.save('X_acc_2_y.npy', np.concatenate(X_acc_y, axis=0))
    np.save('X_acc_2_z.npy', np.concatenate(X_acc_z, axis=0))

    np.save('Y_lin_2_x.npy', np.concatenate(Y_lin_x, axis=0))
    np.save('Y_lin_2_y.npy', np.concatenate(Y_lin_y, axis=0))
    np.save('Y_lin_2_z.npy', np.concatenate(Y_lin_z, axis=0))
    np.save('Y_vel_2_x.npy', np.concatenate(Y_vel_x, axis=0))
    np.save('Y_vel_2_y.npy', np.concatenate(Y_vel_y, axis=0))
    np.save('Y_vel_2_z.npy', np.concatenate(Y_vel_z, axis=0))
    np.save('Y_acc_2_x.npy', np.concatenate(Y_acc_x, axis=0))
    np.save('Y_acc_2_y.npy', np.concatenate(Y_acc_y, axis=0))
    np.save('Y_acc_2_z.npy', np.concatenate(Y_acc_z, axis=0))

def partition_train_test(samples: int, test_size: float, val_size: float, random_seed: int):
    '''
    This function partition the samples into train/val/test by indices
    :param samples: The number of samples in the entire dataset
    :param test_size: The ratio of test samples
    :param val_size: The ratio of validation samples
    :param random_seed: random state seed
    :return: train: training sample indices
             val: validation sample indices
             test: test sample indices
    '''
    idx = [i for i in range(0,samples)]
    train_val, test = train_test_split(idx, test_size=test_size, random_state=random_seed, shuffle=True)
    train, val = train_test_split(train_val,test_size=val_size/(1-test_size), random_state=random_seed,shuffle=True)
    print('Number of training samples: ', len(train))
    print('Number of validation samples: ', len(val))
    print('Number of test samples: ', len(test))
    return train, val, test



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #1.Sliding window partition and data augmentation
    #report_file_name(folders=[Dir_data_1, Dir_data_2,Dir_data_3])
    #process_raw_signals(folders = [Dir_data_1, Dir_data_2,Dir_data_3])

    #2.Build whole set X and Y
    #construct_entire_set(folder=Dir_processed_data)

    #3.Construct the training data, validation data and test data from entire dataset X and Y
    #Warning: The validation data and test data should be from different impacts
    #Pick up 20 samples together so that all the frames from the same record are in the same dataset
    # train_id, val_id, test_id = partition_train_test(samples=163, test_size= 0.15, val_size= 0.15, random_seed = 9001)
    # np.save('Training_2_idx.npy',train_id)
    # np.save('Validation_2_idx.npy',val_id)
    # np.save('Test_2_idx.npy',test_id)

    #4. Additional Test Set
    #report_file_name(folders=[Dir_data_3])
    #process_raw_signals(folders = [Dir_data_3])
    #construct_entire_set(folder=Dir_data_3)

    #5. Merge all data




