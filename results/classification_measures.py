import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy

# This code generates the classification measures throughtout the tables. Different config['specify'] leads to different tables!

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


# (1) COMPUTATION OF CLASSIFICATION MEASURES

# Loop over all classification metrics:
for mtrc in ['PPV', 'NPV', 'TPR', 'TNR', 'ACC']:

    print("METRIC " + mtrc)

    config['metric'] = mtrc

    # Loop over methdos:
    for mthd in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        
        config['method'] = mthd

        # Ground thruth and predictions:
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

            GT[i-1]            = list(np.loadtxt("../dataset/test/GTpointCloud/GTpointCloud" + str(i) + ".txt"))
            GT_info[i-1]       = list(np.loadtxt("../dataset/test/infoPointCloud/infoPointCloud" + str(i) + ".txt"))
            has_unif_noise    += [GT_info[i-1][0]>0]
            has_Gauss_noise   += [GT_info[i-1][3]>0]
            has_undersampling += [GT_info[i-1][6]>0]
            has_missing       += [GT_info[i-1][8]>0]
            has_deformations  += [GT_info[i-1][10]>0]
            
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
        FP = (CM.sum(axis=0) - np.diag(CM)).astype(float)
        FN = (CM.sum(axis=1) - np.diag(CM)).astype(float)
        TP = (np.diag(CM)).astype(float)
        TN = (CM.sum() - (FP + FN + TP)).astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy for each class
        ACC = (TP+TN)/(TP+FP+FN+TN)
        #F1 score
        F1 = (TP)/(TP+1/2*(FP+FN))

        # Print a LateXization of the measures:
        if config['method'] == 'M6':
            if config['metric'] == 'PPV':
                print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{ForestGreen!15}} \\begin{{turn}}{{90}}PPV\end{{turn}}}}}} & {config['method']} & {PPV[0]:.4f} & {PPV[1]:.4f} & {PPV[2]:.4f} & {PPV[3]:.4f} & {PPV[4]:.4f} & {np.mean(PPV):.4f} \\\\")
            if config['metric'] == 'NPV':
                print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{ForestGreen!15}} \\begin{{turn}}{{90}}NPV\end{{turn}}}}}} & {config['method']} & {NPV[0]:.4f} & {NPV[1]:.4f} & {NPV[2]:.4f} & {NPV[3]:.4f} & {NPV[4]:.4f} & {np.mean(NPV):.4f} \\\\")
            if config['metric'] == 'TPR':
                print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{ForestGreen!15}} \\begin{{turn}}{{90}}TPR\end{{turn}}}}}} & {config['method']} & {TPR[0]:.4f} & {TPR[1]:.4f} & {TPR[2]:.4f} & {TPR[3]:.4f} & {TPR[4]:.4f} & {np.mean(TPR):.4f} \\\\")
            if config['metric'] == 'TNR':
                print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{ForestGreen!15}} \\begin{{turn}}{{90}}TNR\end{{turn}}}}}} & {config['method']} & {TNR[0]:.4f} & {TNR[1]:.4f} & {TNR[2]:.4f} & {TNR[3]:.4f} & {TNR[4]:.4f} & {np.mean(TNR):.4f} \\\\")
            if config['metric'] == 'ACC':
                print(f"\multicolumn{{1}}{{|l|}}{{\multirow{{-6}}{{*}}{{\cellcolor{{ForestGreen!15}} \\begin{{turn}}{{90}}ACC\end{{turn}}}}}} & {config['method']} & {ACC[0]:.4f} & {ACC[1]:.4f} & {ACC[2]:.4f} & {ACC[3]:.4f} & {ACC[4]:.4f} & {np.mean(ACC):.4f} \\\\")
        
        else:
            if config['metric'] == 'PPV':
                print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{ForestGreen!15}}}} & {config['method']} & {PPV[0]:.4f} & {PPV[1]:.4f} & {PPV[2]:.4f} & {PPV[3]:.4f} & {PPV[4]:.4f} & {np.mean(PPV):.4f} \\\\")
            if config['metric'] == 'NPV':
                print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{ForestGreen!15}}}} & {config['method']} & {NPV[0]:.4f} & {NPV[1]:.4f} & {NPV[2]:.4f} & {NPV[3]:.4f} & {NPV[4]:.4f} & {np.mean(NPV):.4f} \\\\")
            if config['metric'] == 'TPR':
                print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{ForestGreen!15}}}} & {config['method']} & {TPR[0]:.4f} & {TPR[1]:.4f} & {TPR[2]:.4f} & {TPR[3]:.4f} & {TPR[4]:.4f} & {np.mean(TPR):.4f} \\\\")
            if config['metric'] == 'TNR':
                print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{ForestGreen!15}}}} & {config['method']} & {TNR[0]:.4f} & {TNR[1]:.4f} & {TNR[2]:.4f} & {TNR[3]:.4f} & {TNR[4]:.4f} & {np.mean(TNR):.4f} \\\\")
            if config['metric'] == 'ACC':
                print(f"\multicolumn{{1}}{{|l|}}{{\cellcolor{{ForestGreen!15}}}} & {config['method']} & {ACC[0]:.4f} & {ACC[1]:.4f} & {ACC[2]:.4f} & {ACC[3]:.4f} & {ACC[4]:.4f} & {np.mean(ACC):.4f} \\\\")
                
    print(" ")

print("End of execution")


 
