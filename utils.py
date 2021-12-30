import pickle as pkl
import scipy.sparse
import numpy as np
import pandas as pd
from scipy import sparse as sp
import networkx as nx
from collections import defaultdict
from scipy.stats import uniform
from data import *


def load_data(datadir,mode):
    input_data(datadir,mode = mode)
    PIK = "{}/datasets.dat".format(datadir)
    with open(PIK, "rb") as f:
        objects = pkl.load(f)
    pse_train_data, pse_test_data, pse_val_data, pse_train_label, pse_test_label,pse_val_label, real_st_data, real_st_label,celltypes = tuple(objects)

    pl_train_data = pd.concat([pse_train_data, real_st_data])
    pl_train_label = pd.concat([pse_train_label, real_st_label],)

    pl_train_data = np.array(pl_train_data)
    pse_test_data = np.array(pse_test_data)
    pse_val_data = np.array(pse_val_data)
    pl_train_label = np.array(pl_train_label)
    pse_test_label = np.array(pse_test_label)
    pse_val_label = np.array(pse_val_label)

    #' convert pandas data frame to csr_matrix format
    pl_train_data = scipy.sparse.csr_matrix(pl_train_data.astype('Float64'))
    pse_val_data = scipy.sparse.csr_matrix(pse_val_data.astype('Float64'))
    pse_test_data = scipy.sparse.csr_matrix(pse_test_data.astype('Float64'))

    #' @param M; the number of labeled pseduoST samples in training set
    pse_train_data_len = len(pse_train_data)

    #' 4) get the feature object by combining training, test, valiation sets

    features = sp.vstack((sp.vstack((pl_train_data, pse_val_data)), pse_test_data)).tolil()
    features = preprocess_features(features)

    #' 5) Given cell type, generate three sets of labels with the same dimension


    all_labels = np.concatenate(
        [np.concatenate([pl_train_label, pse_val_label]), pse_test_label])
    all_labels = pd.DataFrame(all_labels)



    #' new label with binary values

    idx_train = range(pse_train_data_len)
    idx_pred = range(pse_train_data_len, len(pl_train_label))
    idx_val = range(len(pl_train_label), len(pl_train_label) + len(pse_val_label))
    idx_test = range(
        len(pl_train_label) + len(pse_val_label),
        len(pl_train_label) + len(pse_val_label) + len(pse_test_label))

    pse_train_mask = sample_mask(idx_train, all_labels.shape[0])
    real_train_mask = sample_mask(idx_pred, all_labels.shape[0])
    val_mask = sample_mask(idx_val, all_labels.shape[0])
    test_mask = sample_mask(idx_test, all_labels.shape[0])

    labels_binary_train = np.zeros(all_labels.shape)
    labels_binary_val = np.zeros(all_labels.shape)
    labels_binary_test = np.zeros(all_labels.shape)

    labels_binary_train[pse_train_mask, :] = all_labels.iloc[pse_train_mask, :]
    labels_binary_val[val_mask, :] = all_labels.iloc[val_mask, :]
    labels_binary_test[test_mask, :] = all_labels.iloc[test_mask, :]

    #' ----- construct adjacent matrix ---------
    #id_graph1一共三列，第一列是index，第二列是pseudo-ST的cell index，第三列是real-ST的cell index
    id_graph1 = pd.read_csv('{}/Linked_graph1.csv'.format(datadir),
                            index_col=0,
                            sep=',')

    ###for ablation(connection)
    # if mode == 'pseudo':
    #     id_graph2 = pd.read_csv('{}/Linked_graph2.csv'.format(datadir),
    #                         sep=',',
    #                         index_col=0)
    # else:
    #     print('load coor files!!!!')
    #     id_graph2 = pd.read_csv('{}/{}_coor.csv'.format(datadir,mode),
    #                             sep=',',
    #                             index_col=0)

    #' --- map index ----
    pse_val_data = pd.DataFrame(pse_val_data)
    pse_test_data = pd.DataFrame(pse_test_data)

    fake1 = np.array([-1] * len(real_st_data.index))
    index_no_real = np.concatenate((pse_train_data.index, fake1, pse_val_data.index,
                             pse_test_data.index)).flatten()

    fake2 = np.array([-1] * len(pse_train_data))
    fake3 = np.array([-1] * (len(pse_val_data) + len(pse_test_data)))
    index_no_pse = np.concatenate((fake2, np.array(real_st_data.index), fake3)).flatten()

    #' ---------------------------------------------
    #'  intra-graph(id_grp1和id_grp2每组点为矩阵中的对角线对称点）
    #' ---------------------------------------------
    #cells的两列全都是real-ST data

    ###for ablation(connection)
    # cells = id_graph2.iloc[:,1:3] + len(pse_train_data)
    # id_grp1 = np.array([0,0])
    # id_grp2 = np.array([0,0])
    #
    # for i in range(len(id_graph2)):
    #     id_grp1 = np.row_stack((id_grp1,[cells.iloc[i,0], cells.iloc[i,1]]))
    #     id_grp2 = np.row_stack((id_grp2,[cells.iloc[i,1], cells.iloc[i,0]]))




    # id_grp1 = np.array([
    #     np.concatenate((np.where(index_no_pse == id_graph2.iloc[i, 2])[0],
    #                     np.where(index_no_pse == id_graph2.iloc[i, 1])[0]))
    #     for i in range(len(id_graph2))
    # ])

    # id_grp2 = np.array([
    #     np.concatenate((np.where(index_no_pse == id_graph2.iloc[i, 1])[0],
    #                     np.where(index_no_pse == id_graph2.iloc[i, 2])[0]))
    #     for i in range(len(id_graph2))
    # ])

    #' ---------------------------------------------
    #'  inter-graph(id_gp1和id_gp2的每组点为矩阵中的对角线对称点）
    #' ---------------------------------------------
    #cells第一列为pseudo，第二列为real
    cells = id_graph1.iloc[:,1:3]
    cells.iloc[:,1] = cells.iloc[:,1] + len(pse_train_data)
    for i in range(len(id_graph1)):
        if id_graph1.iloc[i, 0] < len(pse_train_data):
            id_graph1.iloc[i, 0] = id_graph1.iloc[i, 0] + len(pse_train_data)
        elif (id_graph1.iloc[i, 0] >= len(pse_train_data)) and (id_graph1.iloc[i, 0] < (len(pse_train_data) + len(pse_val_data))):
            id_graph1.iloc[i, 0] = id_graph1.iloc[i, 0] + len(pse_train_data) + len(pse_val_data)
        elif id_graph1.iloc[i, 0] >= (len(pse_train_data) + len(pse_val_data) + len(pse_test_data)):
            id_graph1.iloc[i, 0] = id_graph1.iloc[i, 0] + len(pse_train_data) + len(pse_val_data) + len(pse_test_data)

    id_gp1 = np.array([0,0])
    id_gp2 = np.array([0,0])

    for i in range(len(id_graph1)):
        id_gp1 = np.row_stack((id_gp1,[cells.iloc[i,0], cells.iloc[i,1]]))
        id_gp2 = np.row_stack((id_gp2,[cells.iloc[i,1], cells.iloc[i,0]]))


    # id_gp1 = np.array([
    #     np.concatenate((np.where(index_no_pse == id_graph1.iloc[i, 2])[0],
    #                     np.where(index_no_real == id_graph1.iloc[i, 1])[0]))
    #     for i in range(len(id_graph1))
    # ])
    #
    # id_gp2 = np.array([
    #     np.concatenate((np.where(index_no_real == id_graph1.iloc[i, 1])[0],
    #                     np.where(index_no_pse == id_graph1.iloc[i, 2])[0]))
    #     for i in range(len(id_graph1))
    # ])
    ##matrix为邻接矩阵
    matrix = np.identity(len(all_labels))


    ###for ablation(connection)
    # for i in range(len(id_grp1)):
    #     matrix[id_grp1[i][0],id_grp1[i][1]] = 1
    #     matrix[id_grp2[i][0],id_grp2[i][1]] = 1

    for i in range(len(id_gp1)):
        matrix[id_gp1[i][0],id_gp1[i][1]] = 1
        matrix[id_gp2[i][0],id_gp2[i][1]] = 1
    # for i in range(len(all_labels)):
    #    matrix[i][i] = 0

    adj = graph(matrix)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))

    print("assign input coordinatly....")
    return adj, features, labels_binary_train, labels_binary_val, \
           labels_binary_test, pse_train_mask, real_train_mask, val_mask, test_mask, all_labels, all_labels,celltypes
