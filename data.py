import os
import numpy as np
import random
import pandas as pd
import time as tm
from operator import itemgetter
from sklearn.model_selection import train_test_split
import pickle as pkl
import scipy.sparse
from metrics import *
from gutils import *
from graph import *


#' data preperation
def input_data(DataDir,mode):
    Link_Graph(outputdir='Infor_Data',mode = mode)
    DataPath1 = '{}/Pseudo_ST1.csv'.format(DataDir)
    DataPath2 = '{}/Real_ST2.csv'.format(DataDir)
    LabelsPath1 = '{}/Pseudo_Label1.csv'.format(DataDir)
    LabelsPath2 = '{}/Real_Label2.csv'.format(DataDir)

    #' read the data
    pse_st = pd.read_csv(DataPath1, index_col=0, sep=',')
    real_st_data = pd.read_csv(DataPath2, index_col=0, sep=',')
    pse_st_label = pd.read_csv(LabelsPath1, header=0, index_col=False, sep=',')
    real_st_label = pd.read_csv(LabelsPath2, header=0, index_col=False, sep=',')
    celltypes = pse_st_label.columns

    #pse_st = pse_st.reset_index(drop=True)  #.transpose()
    #real_st_data = real_st_data.reset_index(drop=True)  #.transpose()

    random.seed(123)

    # temD_train, pse_test_data, temL_train, pse_test_label = train_test_split(
    #     pse_st, pse_st_label, test_size=0.01, random_state=1)
    # pse_train_data, pse_val_data, pse_train_label, pse_val_label = train_test_split(
    #     temD_train, temL_train, test_size=0.5, random_state=1)
    pse_train_data, pse_val_data, pse_train_label, pse_val_label = train_test_split(
        pse_st, pse_st_label, test_size=0.1, random_state=1)
    pse_test_data = pse_val_data
    pse_test_label = pse_val_label

    print((pse_train_data.index == pse_train_label.index).all())
    print((pse_test_data.index == pse_test_label.index).all())
    print((pse_val_data.index == pse_val_label.index).all())

    #' save objects

    PIK = "{}/datasets.dat".format(DataDir)
    res = [
        pse_train_data, pse_test_data, pse_val_data, pse_train_label, pse_test_label,
        pse_val_label, real_st_data, real_st_label,celltypes
    ]

    with open(PIK, "wb") as f:
        pkl.dump(res, f)

    print('load data succesfully....')
