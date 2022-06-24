import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import scipy
from sklearn.metrics import confusion_matrix


# This code generates the grouped bar graphs shown in Figure 9. 


# FUNCTIONS FOR LEGENDS

# Function used to generate legends pt. 1
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
    
# Function used to generate legends pt. 2
def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment

# Function used to generate legends pt. 3
def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label:
    padding = 4
    
    # Iterate over angles, values, and labels, to add all of them:
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment:
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text:
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )
        
        
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
    'specify':          0,          # Initialize the perturbation type (check the description above); the code will loop over all types
    'method':           'M1',       # Initialize the method number, the code will loop over all participants
}

# (Empty) lists for the classification measures:
global_PPV = []
global_NPV = []
global_TPR = []
global_TNR = []
global_ACC = []


# (1) COMPUTE CLASSIFICATION MEASURES

# Loop over artifact types:
for artfct in range(1,11):

    config['specify'] = artfct
    
    local_PPV = []
    local_NPV = []
    local_TPR = []
    local_TNR = []
    local_ACC = []

    # Loop over methods:
    for mthd in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        
        config['method'] = mthd

        # Ground truth and predictions:
        GT      = [[] for _ in range(1,config['num_files']+1)]
        GT_info = [[] for _ in range(1,config['num_files']+1)]
        GT_labels = []
        predictions         = [[] for _ in range(1,config['num_files']+1)]
        label_predictions   = []

        # Lists of bools that keep track of which point clouds have some specific artifact:
        has_unif_noise      = []
        has_Gauss_noise     = []
        has_undersampling   = []
        has_missing         = []
        has_deformations    = []

        # Loop over files:
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
        
        # Averaging over files:
        local_PPV += [np.mean(PPV)]
        local_NPV += [np.mean(NPV)]
        local_TPR += [np.mean(TPR)]
        local_TNR += [np.mean(TNR)]
        local_ACC += [np.mean(ACC)]
    
    # Keep track of each method:
    global_PPV += [local_PPV]
    global_NPV += [local_NPV]
    global_TPR += [local_TPR]
    global_TNR += [local_TNR]
    global_ACC += [local_ACC]

# Keep track of each artifact:
global_PPV = np.array(global_PPV)
global_NPV = np.array(global_NPV)
global_TPR = np.array(global_TPR)
global_TNR = np.array(global_TNR)
global_ACC = np.array(global_ACC)
classification_measures = [global_PPV, global_NPV, global_TPR, global_TNR, global_ACC]



# (2) GROUPED BAR GRAPHS

# Loop over classification measures:
for I, titolo in enumerate(["PPV", "NPV", "TPR", "TNR", "ACC"]):

    VALUES = classification_measures[I].reshape((1,60), order='F')[0]
    LABELS = ["A" + str(i) for j in range(1,7) for i in range(1,11)]
    GROUP = ["M" + str(j) for j in range(1,7) for i in range(1,11)]

    # Creating the dataframe:
    d = {'values': VALUES, 'labels': LABELS, 'group': GROUP}
    df = pd.DataFrame(data = d)
    df["labels"] = pd.Categorical(df["labels"], categories=["A" + str(i) for i in range(1,11)])

    df_pivot = pd.pivot_table(
        df,
        values="values",
        index="group",
        columns="labels",
        aggfunc=np.mean
    )

    # Plot a bar chart using the DF:
    ax = df_pivot.plot(kind="bar", width=0.8)
    # Get a Matplotlib figure from the axes object for formatting purposes:
    fig = ax.get_figure()
    # Change the plot dimensions (width, height):
    fig.set_size_inches(4.25, 2)
    # Change the axes labels (here, disabled):
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Title:
    plt.title(titolo)

    # Remove legend:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.get_legend().remove()
    plt.xticks(rotation=0)

    # Save image
    plt.ylim([0.75, 1.01])
    plt.savefig("./images/macro_averages/classification/macro_avgs_" + titolo + ".png")
        


# (3) LEGEND

figsize = (5, 3)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)

# Add the legend from the previous axes:
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(flip(handles, 5), flip(labels, 5), loc='center', ncol=5)

# Hide the axes frame and the x/y labels:
ax_leg.axis('off')
fig_leg.savefig("./images/macro_averages/classification/legend.png")

print("End of execution")


 
