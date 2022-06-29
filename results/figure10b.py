import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.io
import scipy

# This code generates the grouped bar graphs shown in Figure 10 (MFE distance).


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
}

# (1) ERROR COMPUTATION

averages = []

# Loop over artifact types:
for spcf in range(1,11):
    
    config['specify'] = spcf

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

    # Loop over files:
    for i in range(1, config['num_files'] + 1):
        
        GT[i - 1]            = list(np.loadtxt("../dataset/test/GTpointCloud/GTpointCloud" + str(i) + ".txt"))
        GT_info[i - 1]       = list(np.loadtxt("../dataset/test/infoPointCloud/infoPointCloud" + str(i) + ".txt"))
        has_unif_noise      += [GT_info[i - 1][0] > 0]
        has_Gauss_noise     += [GT_info[i - 1][3] > 0]
        has_undersampling   += [GT_info[i - 1][6] > 0]
        has_missing         += [GT_info[i - 1][8] > 0]
        has_deformations    += [GT_info[i - 1][10] > 0]
        
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
            
                GT_labels[i - 1]     = [int(GT[i - 1][0])]
                GT_parameters[i - 1] = GT[i - 1][1:]
                idx2keep            += [i-1]

    M1_MFE = M1_MFE[idx2keep]
    M2_MFE = M2_MFE[idx2keep]
    M3_MFE = M3_MFE[idx2keep]
    M4_MFE = M4_MFE[idx2keep]
    M5_MFE = M5_MFE[idx2keep]
    M6_MFE = M6_MFE[idx2keep]

    # Statistics of the error:
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


# (2) GROUPED BAR GRAPHS

VALUES = np.array(averages).reshape((1,60), order='F')[0]
LABELS = ["T" + str(i) for j in range(1,7) for i in range(1,11)]
GROUP = ["M" + str(j) for j in range(1,7) for i in range(1,11)]

# Creating the dataframe:
d = {'values': VALUES, 'labels': LABELS, 'group': GROUP}
df = pd.DataFrame(data = d)
df["labels"] = pd.Categorical(df["labels"], categories=["T" + str(i) for i in range(1,11)])

df_pivot = pd.pivot_table(
    df,
    values="values",
    index="group",
    columns="labels",
    aggfunc=np.mean
)

# Plot a bar chart using the DF
ax = df_pivot.plot(kind="bar", width=0.8)
# Get a Matplotlib figure from the axes object for formatting purposes
fig = ax.get_figure()
# Change the plot dimensions (width, height)
fig.set_size_inches(4.25, 2)
# Change the axes labels
# ax.set_xlabel("Methods")
# ax.set_ylabel("Performance")
ax.set_xlabel("")
ax.set_ylabel("")
plt.title("MFE")

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_legend().remove()
plt.xticks(rotation=0)

# Use this to show the plot in a new window
plt.savefig("./images/macro_averages/fitting/MFE.png")

print("End of execution")
