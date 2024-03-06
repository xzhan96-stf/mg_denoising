# Mouthguard Denoising Models Based on 1D-CNN
Developed by Xianghao Zhan. Supported by Camlab, Department of Bioengineering, Stanford University and Stanford CS271/BIODS220.

## Environment Requirements
python 3.7, numpy, scipy, torch, sklearn

## How to use the models?
### 1. Download the pre-trained models (ending in ".pt" in the folder "Pretrained Models") and the Python code "Model_predict.py"
### 2. Data Preprocessing
  Prepare the mouthguard kinematics and store the data in .mat file with six variables,
  named as "lin_x", "lin_y", "lin_z", "vel_x", "vel_y", "vel_z": 
  the linear acceleration and angular velocity along the x-(posterior-to-anterior), y-(left-to-right), z-(superior-to-inferior) axis.
  Note that you may need to downsample/upsample the kinematics and then pad/truncate the signals to be 100 points (100ms)!
  The variable should be matrices with the shape (Number of impacts, 100).
  
### 3. Label Preprocessing
  You may also input the label (ATD sensor kinematics) by formulating them under the same rule as step two into a .mat file.
  
### 4. Open a terminal/powershell in the save directory as the "Model_predict.py" and run the following command line.
<code> python3 Model_predict.py <test_filename.mat> <test_label_filename.mat> </code>


## Check the source code and dataset?
Please refer to the folder: Development. 
