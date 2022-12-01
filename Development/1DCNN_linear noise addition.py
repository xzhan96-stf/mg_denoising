import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat,savemat
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

Dir_processed_data = 'H:\\My Drive\\Paper\\MG Denoising\\Data\\Full Set'
Dir_indices = 'C:\\Users\\Lab Admin\\PycharmProjects\\pythonProject'
Dir_modeling_results = 'C:\\Users\\Lab Admin\\PycharmProjects\\pythonProject\\Prediction'
Dir_models = 'C:\\Users\\Lab Admin\\PycharmProjects\\pythonProject\\Model'

def extract_train_test_samples(X, Y):
    '''
    Extract the training samples, test samples and validation samples
    :param X: The entire raw signal datasets
    :param Y: The entire clean signal datasets
    :param train_id: The list of training sample indices
    :param val_id: The list of validation sample indices
    :param test_id: The list of test sample indices
    :return: train, val, test
    '''
    os.chdir(Dir_indices)
    train_id = np.load('Training_2_idx.npy')
    val_id = np.load('Validation_2_idx.npy')
    test_id = np.load('Test_2_idx.npy')
    train_idx = np.array([list(range(i*20,i*20+20)) for i in train_id]).reshape(-1,)
    val_idx = np.array([list(range(i*20,i*20+20)) for i in val_id]).reshape(-1,)
    test_idx = np.array([list(range(i*20,i*20+20)) for i in test_id]).reshape(-1,)
    train_X, train_Y =  X[train_idx], Y[train_idx]
    val_X, val_Y = X[val_idx], Y[val_idx]
    test_X, test_Y = X[test_idx], Y[test_idx]
    return train_X, val_X, test_X, train_Y, val_Y, test_Y

def X_preprocessing(X, method:str):
    if method == 'STD':
        ss = StandardScaler()
        X = ss.fit_transform(X)

    return (X, ss)

class Dataset(torch.utils.data.Dataset):
    '''
    Prepare the denoising dataset
    '''

    def __init__(self,X,y, scale_data=False):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X, ss = X_preprocessing(X,'STD')
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

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

def plot_trainval_loss(loss_recorder,val_loss_recorder, title, spec):
    plt.figure(figsize=(9, 6))
    plt.plot(loss_recorder)
    plt.plot(val_loss_recorder)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid()
    plt.title(title)
    plt.savefig(spec+'.png',dpi=600,bbox_inches='tight')
    plt.show()

def compute_validation_loss(valloader,model,loss_function):
    for i, data in enumerate(valloader, 0):
        # Get inputs and outputs
        input, target = data
        input, target = input.float(), target.float()
        output = model(input) #This is the noise to be added
    return loss_function(output, target-input).item()

def evaluate(dataloader,model):
    start = time.time()
    for i, data in enumerate(dataloader, 0):
        # Get inputs and outputs
        input, target = data
        input, target = input.float(), target.float()
        output = model(input) #The noise to be added to the input X
        output = input+output #The real output: input X + noise
    end = time.time()
    target = torch.squeeze(target).detach().numpy()
    output = torch.squeeze(output).detach().numpy()
    prediction_time = -(start-end)/output.shape[0]
    MAE = mean_absolute_error(output,target)
    RMSE = np.sqrt(mean_squared_error(output,target))
    R2 = r2_score(target,output)
    return MAE,RMSE,R2,target,output,prediction_time

def evaluate_raw(output,target):
    output = output.squeeze()
    target = target.squeeze()
    MAE = mean_absolute_error(output,target)
    RMSE = np.sqrt(mean_squared_error(output,target))
    R2 = r2_score(target,output)
    return MAE,RMSE,R2

if __name__ == '__main__':
    #Define task
    target = 'lin_X'
    learning_rate = 5e-3
    channels = [20,40,80]
    kernel_size = 10
    epochs = 1200
    batchsize= 2000
    init_learning_rate = learning_rate

    #Set fixed random number seed
    torch.manual_seed(9001)


    #Load dataset
    os.chdir(Dir_processed_data)
    X = np.load('X_'+target+'.npy').reshape(-1,1,100)
    Y = np.load('Y_'+target+'.npy').reshape(-1,1,100)

    #Extrac train val test
    train_X, val_X, test_X, train_Y, val_Y, test_Y = extract_train_test_samples(X,Y)

    #Data augmentation with symmetry consideration
    train_X = np.concatenate((train_X,-train_X),axis=0)
    train_Y = np.concatenate((train_Y,-train_Y),axis=0)
    print(train_X.shape)

    train_add_y = train_Y-train_X #The difference between the train_y and train_x
    val_add_y = val_Y - val_X
    test_add_y = test_Y - test_X

    train_dataset = Dataset(X=train_X,y=train_add_y)
    train_dataset_test = Dataset(X=train_X, y=train_Y)
    val_dataset = Dataset(X=val_X,y=val_Y)
    test_dataset = Dataset(X=test_X,y=test_Y)

    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=4800,shuffle=True, num_workers=1)
    trainloader_test = torch.utils.data.DataLoader(train_dataset_test, batch_size=2400, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1000,shuffle=False,num_workers=1)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=1)

    #Initialize the MLP
    convautoencoder = ConvAutoencoder(channels=channels,kernel_size=kernel_size)

    #Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(convautoencoder.parameters(),lr=learning_rate,weight_decay=1e-2)

    #Running the training loop
    loss_recorder = []
    val_loss_recorder = []
    for epoch in range(0,epochs):
        if epoch % 50 == 0:
            learning_rate = learning_rate*0.9
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        #Print epoch
        print(f'Starting epoch {epoch+1}')

        #Set current loss value
        current_loss = 0.0

        #Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader,0):

            #Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            optimizer.zero_grad()

            #Perform forward pass
            outputs = convautoencoder(inputs)

            #Compute loss
            loss = loss_function(outputs,targets)

            #Perform backward pass
            loss.backward()

            #Perform optimization
            optimizer.step()

            #Print statistics
            current_loss += loss.item()

            if i % batchsize == 0:
                print('Loss after mini-batch %5d: %.3f' % (i+1, current_loss))
                loss_recorder.append(current_loss)
                current_loss = 0.0

        #Training Process is complete, start the validation process
        val_loss_recorder.append(compute_validation_loss(valloader,convautoencoder,loss_function))


    print('The training process has finished')

    #Model test on the train/val/test sets and output metrics
    MAL,RMSL,R2L,y_train,y_train_pred,_ = evaluate(trainloader_test,convautoencoder)
    MAV,RMSV,R2V,y_val,y_val_pred,_ = evaluate(valloader,convautoencoder)
    MAE,RMSE,R2E,y_test,y_test_pred,prediction_time = evaluate(testloader,convautoencoder)
    MAX, RMSX, R2X = evaluate_raw(test_X, test_Y)
    print('Prediction time: %.3f(%.3f)', prediction_time)
    print("MAL: %.3f(%.3f)", MAL)
    print("MAV: %.3f(%.3f)", MAV)
    print("MAE: %.3f(%.3f)", MAE)
    print("MAX: %.3f(%.3f)", MAX)
    print("RMSL: %.3f(%.3f)", RMSL)
    print("RMSV: %.3f(%.3f)", RMSV)
    print("RMSE: %.3f(%.3f)", RMSE)
    print("RMSX: %.3f(%.3f)", RMSX)
    print("R2_train: %.3f(%.3f)", R2L)
    print("R2_val: %.3f(%.3f)", R2V)
    print("R2_test: %.3f(%.3f)", R2E)
    print("R2_raw X: %.3f(%.3f)", R2X)

    os.chdir(Dir_modeling_results)
    layers = '_'.join([str(i) for i in channels])
    specs = 'ConvAutoencoder_noiselinear_fullset_AUG_' + target + '_channels_' + layers + '_kernel_size_' + str(kernel_size) + '_initial lr_' + str(init_learning_rate) + '_epoch_' + str(epochs) + 'batchnorm_l2_1e-2'
    plot_trainval_loss(loss_recorder, val_loss_recorder, title='Train Val Loss',spec=specs)
    name = specs +'.mat'
    savemat(name,{'X_train': train_X, 'Y_train': y_train, 'Y_train_pred': y_train_pred, 'MAL': MAL, 'RMSL': RMSL, 'R2L': R2L,
             'X_val': val_X, 'Y_val': y_val, 'Y_val_pred': y_val_pred, 'MAV': MAV, 'RMSV': RMSV, 'R2V': R2V,
             'X_test': test_X, 'Y_test': y_test, 'Y_test_pred': y_test_pred, 'MAE': MAE, 'RMSE': RMSE, 'R2E': R2E,
             'Prediction Time': prediction_time})
    print('Predictions statistics saved to disk!')

    #Save models
    os.chdir(Dir_models)
    torch.save({
        'epoch': epochs,
        'model_state_dict':convautoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_recorder[-1]}, specs+'.pt')
    print('Model saved to disk!')
