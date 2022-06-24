import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.io
import scipy

# This code generates the MFE measures throughtout the tables. Different config['specify'] leads to different tables!


# (0) PARAMETERS

# List of possible choices for config['specify'], which specifies which kind of perturbation (if any) to consider.
# config['specify'] =   0   (all pointclouds)
#                       1   (clean)
#                       2   (uniform noise)
#                       3   (Gaussian noise)
#                       4   (undersampling)
#                       5   (missing data)
#                       6   (uniform noise + undersampling)
#                       7   (Gaussian noise + undersampling)
#                       8   (uniform noise + missing data)
#                       9   (Gaussian noise + missing data)
#                      10   (small deformations)
config = {
    'num_files':        925,        # Number of point clouds, do NOT change
    'specify':          0,          # Initialize the perturbation type (check the description above)
}


# (1) COMPUTATION OF MDE DISTANCES

averages = []

# Ground truth and predictions:
GT              = [[] for _ in range(1, config['num_files'] + 1)]
GT_labels       = [[] for _ in range(1, config['num_files'] + 1)]
GT_parameters   = [[] for _ in range(1, config['num_files'] + 1)]
GT_info         = [[] for _ in range(1, config['num_files'] + 1)]
has_unif_noise      = []
has_Gauss_noise     = []
has_undersampling   = []
has_missing         = []
has_deformations    = []
idx2keep            = []

M1_MFE = np.squeeze(scipy.io.loadmat("./methods/" + 'M1' + "/MFE.mat")['mfe'])
M2_MFE = np.squeeze(scipy.io.loadmat("./methods/" + 'M2' + "/MFE.mat")['mfe'])
M3_MFE = np.squeeze(scipy.io.loadmat("./methods/" + 'M3' + "/MFE.mat")['mfe'])
M4_MFE = np.squeeze(scipy.io.loadmat("./methods/" + 'M4' + "/MFE.mat")['mfe'])
M5_MFE = np.squeeze(scipy.io.loadmat("./methods/" + 'M5' + "/MFE.mat")['mfe'])
M6_MFE = np.squeeze(scipy.io.loadmat("./methods/" + 'M6' + "/MFE.mat")['mfe'])

# Loop over all point clouds:
for i in range(1, config['num_files'] + 1):
    
    GT[i - 1]            = list(np.loadtxt("../benchmark/dataset/GTpointCloud/GTpointCloud" + str(i) + ".txt"))
    GT_info[i - 1]       = list(np.loadtxt("../benchmark/dataset/infoPointCloud/infoPointCloud" + str(i) + ".txt"))
    has_unif_noise      += [GT_info[i - 1][0] > 0]
    has_Gauss_noise     += [GT_info[i - 1][3] > 0]
    has_undersampling   += [GT_info[i - 1][6] > 0]
    has_missing         += [GT_info[i - 1][8] > 0]
    has_deformations    += [GT_info[i - 1][10] > 0]
    
    # Check specific perturbation type:
    if config['specify']==0 or \
        (config['specify']==1 and has_unif_noise[-1] + has_Gauss_noise[-1] + has_undersampling[-1] + has_missing[-1] + has_deformations[-1] == 0) or \
        (config['specify']==2 and has_unif_noise[-1] == 1) or \
        (config['specify']==3 and has_Gauss_noise[-1] == 1) or \
        (config['specify']==4 and has_undersampling[-1] == 1) or \
        (config['specify']==5 and has_missing[-1] == 1) or \
        (config['specify']==6 and has_unif_noise[-1] == 1 and has_undersampling[-1] == 1) or \
        (config['specify']==7 and has_Gauss_noise[-1] == 1 and has_undersampling[-1] == 1) or \
        (config['specify']==8 and has_unif_noise[-1] == 1 and has_missing[-1] == 1) or \
        (config['specify']==9 and has_Gauss_noise[-1] == 1 and has_missing[-1] == 1) or \
        (config['specify']==10 and has_deformations[-1] == 1):
        
            GT_labels[i - 1]     = [int(GT[i - 1][0])]
            GT_parameters[i - 1] = GT[i - 1][1:]
            idx2keep            += [i-1]
M1_MFE = M1_MFE[idx2keep]
M2_MFE = M2_MFE[idx2keep]
M3_MFE = M3_MFE[idx2keep]
M4_MFE = M4_MFE[idx2keep]
M5_MFE = M5_MFE[idx2keep]
M6_MFE = M6_MFE[idx2keep]

# Print a LateXization of the measures:
curr_descr = pd.DataFrame(M1_MFE).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M1 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M2_MFE).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M2 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M3_MFE).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M3 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M4_MFE).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M4 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M5_MFE).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M5 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M6_MFE).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{BlueViolet!20}} \\begin{{turn}}{{90}}MFE\end{{turn}}}}}} & M6 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e} \\\\")
print("\\hline")

local_averages = []

curr_descr = pd.DataFrame(M1_MFE).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M2_MFE).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M3_MFE).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M4_MFE).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M5_MFE).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M6_MFE).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]

averages += [local_averages]

print("End of execution")
