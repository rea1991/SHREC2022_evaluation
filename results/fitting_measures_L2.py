import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy

# This code generates the L2 measures throughtout the tables. Different config['specify'] leads to different tables!


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


# (1) FUNCTIONS USED IN THE ERROR COMPUTATION

# Check signs for the i-th entries of vec1 and vec2:
def recover_unicity_i(vec1, vec2, i):
    if vec1[i] * vec2[i] < 0:
        vec2 = - vec2
        return True, vec1, vec2
    elif vec1[i] * vec2[i] > 0:
        return True, vec1, vec2
    else:
        return False, vec1, vec2


# Recover unicity for unit vectors:
def recover_unicity(vec1, vec2):
    for i in range(3):
        changed_or_not, vec1, vec2 = recover_unicity_i(vec1, vec2, i)
        if changed_or_not:
            return vec1, vec2

    return list(vec1), list(vec2)


# Compute L2 error:
def L2_error(GT_labels, GT_parameters, participant):
    
    # Predictions:
    predictions             = [[] for _ in range(1, config['num_files'] + 1)]
    label_predictions       = []
    parameter_predictions   = [[] for _ in range(1, config['num_files'] + 1)]
    
    # Load information:
    for i in range(1, config['num_files'] + 1):
        predictions[i - 1] = list(
            np.loadtxt("./methods/" + participant + "/prediction_results/pointCloud" + str(i) + "_prediction.txt"))
        label_predictions           += [int(predictions[i - 1][0])]
        parameter_predictions[i - 1] = predictions[i - 1][1:]

    # List of errors:
    errors          = []
    segments4errors = []
    
    # Loop over all files:
    for i in range(config['num_files']):
    
        if len(GT_labels[i])>0:
        
            GT_label = GT_labels[i][0]
            label_prediction = label_predictions[i]

            # Keep only true positives
            if GT_label == label_predictions[i]:
            
                segments4errors += [i + 1]

                # TORI
                if GT_label == 5:

                    # First minor radius, then major radius:
                    if parameter_predictions[i][0] > parameter_predictions[i][1]:
                        parameter_predictions[i][0:2] = parameter_predictions[i][0:2][::-1]
                    if GT_parameters[i][0] > GT_parameters[i][1]:
                        GT_parameters[i][0:2] = GT_parameters[i][0:2][::-1]

                    # Restoring uniqueness:
                    GT_parameters[i][2:5], parameter_predictions[i][2:5] = \
                        recover_unicity(np.array(GT_parameters[i][2:5]), np.array(parameter_predictions[i][2:5]))

                    # Errors:
                    errors += [np.linalg.norm(np.array(GT_parameters[i]) - np.array(parameter_predictions[i]))]

                # CYLINDERS
                elif GT_label == 2:

                    # Restoring uniqueness:
                    GT_parameters[i][1:4], parameter_predictions[i][1:4] = \
                        recover_unicity(np.array(GT_parameters[i][1:4]), np.array(parameter_predictions[i][1:4]))

                    # Errors:
                    errors += [np.linalg.norm(np.array(GT_parameters[i][0:4]) - np.array(parameter_predictions[i][0:4]))]

                # CONES
                elif GT_label == 4:

                    # Restoring uniqueness:
                    GT_parameters[i][1:4], parameter_predictions[i][1:4] = \
                        recover_unicity(np.array(GT_parameters[i][1:4]), np.array(parameter_predictions[i][1:4]))

                    # Errors:
                    errors += [np.linalg.norm(np.array(GT_parameters[i]) - np.array(parameter_predictions[i]))]

                # SPHERES
                elif GT_label == 3:

                    # Errors:
                    errors += [np.linalg.norm(np.array(GT_parameters[i]) - np.array(parameter_predictions[i]))]

                # PLANES
                else:

                    # Restoring uniqueness:
                    GT_parameters[i][0:3], parameter_predictions[i][0:3] = \
                        recover_unicity(np.array(GT_parameters[i][0:3]), np.array(parameter_predictions[i][0:3]))

                    # Errors:
                    errors += [np.linalg.norm(np.array(GT_parameters[i][0:3]) - np.array(parameter_predictions[i][0:3]))]

    return errors, segments4errors


# (2) COMPUTATION OF THE L2 ERROR

averages = []

# Ground thruth and predictions:
GT              = [[] for _ in range(1, config['num_files'] + 1)]
GT_labels       = [[] for _ in range(1, config['num_files'] + 1)]
GT_parameters   = [[] for _ in range(1, config['num_files'] + 1)]
GT_info         = [[] for _ in range(1, config['num_files'] + 1)]
has_unif_noise      = []
has_Gauss_noise     = []
has_undersampling   = []
has_missing         = []
has_deformations    = []

# Loop over all files:
for i in range(1, 925 + 1):
    
    GT[i - 1]            = list(np.loadtxt("../benchmark/dataset/GTpointCloud/GTpointCloud" + str(i) + ".txt"))
    GT_info[i - 1]       = list(np.loadtxt("../benchmark/dataset/infoPointCloud/infoPointCloud" + str(i) + ".txt"))
    has_unif_noise      += [GT_info[i - 1][0] > 0]
    has_Gauss_noise     += [GT_info[i - 1][3] > 0]
    has_undersampling   += [GT_info[i - 1][6] > 0]
    has_missing         += [GT_info[i - 1][8] > 0]
    has_deformations    += [GT_info[i - 1][10] > 0]
    
    # Check the chosen perturbation type:
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

# Actual L2 computation:
M1_L2, M1_idx = L2_error(GT_labels, GT_parameters, 'M1')
M2_L2, M2_idx = L2_error(GT_labels, GT_parameters, 'M2')
M3_L2, M3_idx = L2_error(GT_labels, GT_parameters, 'M3')
M4_L2, M4_idx = L2_error(GT_labels, GT_parameters, 'M4')
M5_L2, M5_idx = L2_error(GT_labels, GT_parameters, 'M5')
M6_L2, M6_idx = L2_error(GT_labels, GT_parameters, 'M6')
M1_L2 = np.array(M1_L2)
M2_L2 = np.array(M2_L2)
M3_L2 = np.array(M3_L2)
M4_L2 = np.array(M4_L2)
M5_L2 = np.array(M5_L2)
M6_L2 = np.array(M6_L2)

# Print a LateXization of the measures:
curr_descr = pd.DataFrame(M1_L2).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M1 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M2_L2).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M2 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M3_L2).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M3 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M4_L2).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M4 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M5_L2).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{BlueViolet!20}}}} & M5 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e}\\\\")
curr_descr = pd.DataFrame(M6_L2).describe()
print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{BlueViolet!20}} \\begin{{turn}}{{90}}$L_2$\end{{turn}}}}}} & M6 & {curr_descr.iloc[4,0]:.2e} & {curr_descr.iloc[5,0]:.2e} & {curr_descr.iloc[6,0]:.2e} & {curr_descr.iloc[1,0]:.2e} & {curr_descr.iloc[2,0]:.2e} \\\\")
print("\\hline")

local_averages = []

curr_descr = pd.DataFrame(M1_L2).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M2_L2).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M3_L2).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M4_L2).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M5_L2).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]
curr_descr = pd.DataFrame(M6_L2).describe()
local_averages += [math.log(curr_descr.iloc[1,0])]

averages += [local_averages]


print("End of execution")
