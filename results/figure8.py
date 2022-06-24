import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy

# This code generates the confusion matrices shown in Figure 8. It optionally provides the confusion matrices for specific perturbation types, if config['specify'] is changed as specified below.


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
    'specify':          0,          # Check the description above (do NOT change if you want to reproduce Figure 8)
    'method':           'M1',       # Initialize the method number, the code will loop over all participants
}

# (1) CONFUSION MATRIX

# Loop over all methods:
for mthd in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
    
    config['method'] = mthd

    # Ground truth and predictions:
    GT      = [[] for _ in range(1,config['num_files']+1)]
    GT_info = [[] for _ in range(1,config['num_files']+1)]
    GT_labels = []

    predictions         = [[] for _ in range(1,config['num_files']+1)]
    label_predictions   = []
    has_unif_noise      = []
    has_Gauss_noise     = []
    has_undersampling   = []
    has_missing         = []
    has_deformations    = []

    # Loop over all point clouds:
    for i in range(1,config['num_files']+1):

        GT[i-1]            = list(np.loadtxt("../benchmark/dataset/GTpointCloud/GTpointCloud" + str(i) + ".txt"))
        GT_info[i-1]       = list(np.loadtxt("../benchmark/dataset/infoPointCloud/infoPointCloud" + str(i) + ".txt"))
        has_unif_noise    += [GT_info[i-1][0]>0]
        has_Gauss_noise   += [GT_info[i-1][3]>0]
        has_undersampling += [GT_info[i-1][6]>0]
        has_missing       += [GT_info[i-1][8]>0]
        has_deformations  += [GT_info[i-1][10]>0]
        
        # Check for the selected perturbation type:
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
            
            GT_labels += [int(GT[i-1][0])]
            predictions[i-1] = list(np.loadtxt("./methods/" + config['method'] + "/prediction_results/pointCloud" + str(i) + "_prediction.txt"))
            label_predictions += [int(predictions[i-1][0])]
        

    # Confusion matrix:
    CM = confusion_matrix(GT_labels, label_predictions)
    df_CM = pd.DataFrame(CM, index = [i for i in ['planes', 'cylinders', 'spheres', 'cones', 'tori']],
                      columns = [i for i in ['planes', 'cylinders', 'spheres', 'cones', 'tori']])
    df_CM.astype('int32').dtypes
    plt.figure(figsize = (5,4))
    sn.heatmap(df_CM, annot=True, fmt='d', cbar=False)
    plt.savefig("./images/confusion_matrices/" + config['method'] + ".png")

print("End of execution")


 
