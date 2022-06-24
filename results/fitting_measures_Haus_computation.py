import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import directed_hausdorff

from scipy.spatial import KDTree


config = {
    'num_files':        925,
    'method':           'M1'
    
}



# GROUND TRUTH AND PREDICTIONS
GT              = [[] for _ in range(1, config['num_files'] + 1)]
GT_labels       = []
GT_info         = [[] for _ in range(1, config['num_files'] + 1)]
has_unif_noise      = []
has_Gauss_noise     = []
has_undersampling   = []
has_missing         = []
has_deformations    = []

HausPCSup = []
HausXSup  = []
for i in range(1, config['num_files'] + 1):

    print("MODEL " + str(i))
    
    GT[i - 1]            = list(np.loadtxt("../benchmark/dataset/GTpointCloud/GTpointCloud" + str(i) + ".txt"))
    GT_info[i - 1]       = list(np.loadtxt("../benchmark/dataset/infoPointCloud/infoPointCloud" + str(i) + ".txt"))
    has_unif_noise      += [GT_info[i - 1][0] > 0]
    has_Gauss_noise     += [GT_info[i - 1][3] > 0]
    has_undersampling   += [GT_info[i - 1][6] > 0]
    has_missing         += [GT_info[i - 1][8] > 0]
    has_deformations    += [GT_info[i - 1][10] > 0]
    GT_labels           += [int(GT[i - 1][0])]
    
    
    X   = np.loadtxt("./methods/" + config['method'] + "/4Hausdorff/X/X" + str(i) + ".txt", delimiter=',')
    sup = np.loadtxt("./methods/" + config['method'] + "/4Hausdorff/sup/sup" + str(i) + ".txt", delimiter=',')
    HausXSup  += [directed_hausdorff(X, sup)[0]]
        
np.savetxt("./methods/" + config['method'] + "/Hausdorff.txt", list(HausXSup))


#print(pd.DataFrame(HausXSup).describe())
