import numpy as np
from scipy.io import loadmat, savemat
import os
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat,savemat
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

class ConvAutoencoder(nn.Module):
    def __init__(self,channels=[16,32,64],kernel_size=10):
        super(ConvAutoencoder,self).__init__()

        #Encoder
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=channels[0],kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=channels[0],out_channels=channels[1],kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=channels[1],out_channels=channels[2],kernel_size=kernel_size)

        #Decoder
        self.t_conv1 = nn.ConvTranspose1d(in_channels=channels[2],out_channels=channels[1],kernel_size=kernel_size)
        self.t_conv2 = nn.ConvTranspose1d(in_channels=channels[1],out_channels=channels[0],kernel_size=kernel_size)
        self.t_conv3 = nn.ConvTranspose1d(in_channels=channels[0], out_channels=1,kernel_size=kernel_size)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = self.t_conv3(x)
        return x
class Dataset(torch.utils.data.Dataset):
    '''
    Prepare the denoising dataset
    '''

    def __init__(self,X,y, scale_data=False):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def evaluate_pointwise(y_true,y_pred):
    MAE = mean_absolute_error(y_true,y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
    return MAE, RMSE

def evaluate_peak(y_true, y_pred):
    MAE = mean_absolute_error(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1))
    RMSE = np.sqrt(mean_squared_error(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1)))
    R2 = r2_score(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1))
    PR = pearsonr(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1))[0]
    SR = spearmanr(np.max(np.abs(y_true),axis=1), np.max(np.abs(y_pred),axis=1))[0]
    return MAE, RMSE, R2, PR, SR

def compute_SNR_out(y_true, y_pred):
    SNR_out = 10*np.log10(np.sum(np.square(y_true),axis=1)/np.sum(np.square(y_true-y_pred),axis=1)) #(N,)
    return np.mean(SNR_out)

def compute_SNR_in(y_true, raw):
    SNR_in = 10*np.log10(np.sum(np.square(y_true),axis=1)/np.sum(np.square(y_true-raw),axis=1)) #(N,)
    return np.mean(SNR_in)

def compute_magnitude(x,y,z):
    return np.sqrt(np.square(x)+np.square(y)+np.square(z))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #1. Define the pretrained_model_name and the filename and directory for the kinematics to be predicted
    # Define the directory and filenames
    current_dir = os.getcwd()
    Dir_model = current_dir + '/Pretrained Models'
    Dir_test_kinematics = current_dir
    
    #Get the filenames for the test data and test labels
    args = sys.argv[1:]
    if len(args) == 2: # Both the test impacts and the test labels are given
        test_filename = str(args[0])
        test_label_filename = str(args[1])
    else: # Only the test impact kinematics are given, no labels
        test_filename = str(args[0])
        test_label_filename = None
        
    outcomes = ['lin_x','lin_y','lin_z','vel_x','vel_y','vel_z']
    structures = [[16,32,64], [20,40,80], [32,64,128], [16,32,64], [16,32,64], [20,40,80]]
    strategies = ['direct','noise' ,'direct', 'direct', 'direct', 'noise'] # direct modeling (direct) or linear noise addition (noise)

    for idx, outcome in enumerate(outcomes):
        #2. Define the model without state dict loaded
        channels = structures[idx]
        kernel_size = 10
        strategy = strategies[idx]

        conv = ConvAutoencoder(channels=channels,kernel_size=kernel_size)

        #3. Load trained model
        os.chdir(Dir_model)
        checkpoint = torch.load(f = outcome + '.pt')
        conv.load_state_dict(checkpoint['model_state_dict'])

        os.chdir(Dir_test_kinematics)
        test_x = loadmat(test_filename)[outcome].reshape(-1,1,100)

        test_dataset = Dataset(X=test_x,y=np.zeros((test_x.shape[0],1)))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_x.shape[0], shuffle=False, num_workers=1)


        if strategy == "direct":
            #4. Make Predictions (1D-CNN)
            for i, data in enumerate(testloader, 0):
                # Get inputs and outputs
                inputs, _ = data
                inputs = inputs.float()
                output = conv(inputs)
            output = torch.squeeze(output).detach().numpy()
            inputs = torch.squeeze(inputs).detach().numpy()
            assert output.shape == inputs.shape
        else:
            # 4. Make Predictions (1D-CNN Noise Linear Addition)
            for i, data in enumerate(testloader, 0):
                # Get inputs and outputs
                inputs, _ = data
                inputs = inputs.float()
                output = conv(inputs) + inputs # Linear noise addition
            output = torch.squeeze(output).detach().numpy()
            inputs = torch.squeeze(inputs).detach().numpy()
            assert output.shape == inputs.shape

        #5. Evaluate the RMSE, PAE between the denoised/raw MG measurement if the ground truth kinematics are present
        if test_label_filename != None:
            test_label = loadmat(test_filename)[outcome].reshape(-1,100)

            mean_MAE, mean_RMSE = evaluate_pointwise(test_label, output)
            mean_MAE_baseline, mean_RMSE_baseline = evaluate_pointwise(test_label, inputs)
            MAE, RMSE, R2, PR, SR = evaluate_peak(test_label,output)
            MAE_baseline, RMSE_baseline, R2_baseline, PR_baseline, SR_baseline = evaluate_peak(test_label, inputs)
            SNR_in = compute_SNR_in(test_label, inputs)
            SNR_out = compute_SNR_out(test_label,output)

            print("Channel to be denoised: ", outcome)
            print(';'.join([str(round(mean_MAE,3)), str(round(mean_RMSE,3)),  str(round(MAE,3)),str(round(RMSE,3)),str(round(R2,3)),str(round(SNR_out,3)), str(round(PR, 3)), str(round(SR, 3))]))
            print(';'.join([str(round(mean_MAE_baseline,3)), str(round(mean_RMSE_baseline,3)), str(round(MAE_baseline,3)),str(round(RMSE_baseline,3)),str(round(R2_baseline,3)),str(round(SNR_in,3)), str(round(PR_baseline, 3)), str(round(SR_baseline, 3))]))

        #6. Save prediction
        os.chdir(Dir_test_kinematics)
        savemat(outcome + '.mat',{'Raw MG':inputs,'Denoised MG': output})



