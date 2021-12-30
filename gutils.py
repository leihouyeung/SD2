import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KDTree
from metrics import *
from keras.models import Model
from keras.layers import Dense, Input
import tensorflow as tf

#' @param num.cc Number of canonical vectors to calculate
#' @param seed.use Random seed to set.
#' @importFrom SVD
# num_cc 是最终图中每个节点的特征向量长度，
def embed(data1, data2, num_cc=20):
    random.seed(123)
    object1 = Scale(data1)
    object2 = Scale(data2)
    mat3 = np.matmul(np.matrix(object1).transpose(), np.matrix(object2))
    a = SVD(mat=mat3, num_cc=int(num_cc))
    embeds_data = np.concatenate((a[0], a[1]))
    ind = np.where(
        [embeds_data[:, col][0] < 0 for col in range(embeds_data.shape[1])])[0]
    embeds_data[:, ind] = embeds_data[:, ind] * (-1)

    embeds_data = pd.DataFrame(embeds_data)
    embeds_data.index = np.concatenate(
        (np.array(data1.columns), np.array(data2.columns)))
    embeds_data.columns = ['D_' + str(i) for i in range(num_cc)]
    d = a[2]
    #' d = np.around(a[2], 3)  #.astype('int')
    return embeds_data, d


def Embed(data_use1, data_use2, features, count_names, num_cc):
    features = checkFeature(data_use1, features)
    features = checkFeature(data_use2, features)
    data1 = data_use1.loc[features, ]
    data2 = data_use2.loc[features, ]
    embed_results = embed(data1=data1, data2=data2, num_cc=num_cc)
    cell_embeddings = np.matrix(embed_results[0])
    combined_data = data1.merge(data2,
                                left_index=True,
                                right_index=True,
                                how='inner')
    new_data1 = combined_data.loc[count_names, ].dropna()
    # loadings=loadingDim(new.data1,cell.embeddings)
    loadings = pd.DataFrame(np.matmul(np.matrix(new_data1), cell_embeddings))
    loadings.index = new_data1.index
    return embed_results, loadings


def checkFeature(data_use, features):
    data1 = data_use.loc[features, ]
    feature_var = data1.var(1)
    Var_features = features[np.where(feature_var != 0)[0]]
    return Var_features

#dist输出n个点和所有n个点的top-k的最小距离，ind输出输出每一个n个点和所有n个点的top-k的最小距离的点的index
def kNN(data, k, query=None):
    tree = KDTree(data)
    if query is None:
        query = data
    dist, ind = tree.query(query, k)
    return dist, ind


#输出cell1和cell2中自己与自己的kNN结果、和对方kNN的结果，k表示取top-k个距离最近的点
#' @param cell_embedding : pandas data frame
def KNN(cell_embedding, cells1, cells2, k):
    embedding_cells1 = cell_embedding.loc[cells1, ]
    embedding_cells2 = cell_embedding.loc[cells2, ]
    nnaa = kNN(embedding_cells1, k=k + 1)
    nnbb = kNN(embedding_cells2, k=k + 1)
    nnab = kNN(data=embedding_cells2, k=k, query=embedding_cells1)
    nnba = kNN(data=embedding_cells1, k=k, query=embedding_cells2)
    return nnaa, nnab, nnba, nnbb, cells1, cells2

#mutual nearest neighbours，输出了两列点，每一行为两个点的index，相互连线
def MNN(neighbors, colnames, num):
    max_nn = np.array([neighbors[1][1].shape[1], neighbors[2][1].shape[1]])
    if ((num > max_nn).any()):
        num = np.min(max_nn)
        # convert cell name to neighbor index
    cells1 = colnames
    cells2 = colnames
    nn_cells1 = neighbors[4]
    nn_cells2 = neighbors[5]
    cell1_index = [
        list(nn_cells1).index(i) for i in cells1 if (nn_cells1 == i).any()
    ]
    cell2_index = [
        list(nn_cells2).index(i) for i in cells2 if (nn_cells2 == i).any()
    ]
    ncell = range(neighbors[1][1].shape[0])
    ncell = np.array(ncell)[np.in1d(ncell, cell1_index)]
    # initialize a list
    mnn_cell1 = [None] * (len(ncell) * 5)
    mnn_cell2 = [None] * (len(ncell) * 5)
    idx = -1
    for cell in ncell:
        neighbors_ab = neighbors[1][1][cell, 0:5]
        mutual_neighbors = np.where(
            neighbors[2][1][neighbors_ab, 0:5] == cell)[0]
        for i in neighbors_ab[mutual_neighbors]:
            idx = idx + 1
            mnn_cell1[idx] = cell
            mnn_cell2[idx] = i
    mnn_cell1 = mnn_cell1[0:(idx + 1)]
    mnn_cell2 = mnn_cell2[0:(idx + 1)]
    import pandas as pd
    mnns = pd.DataFrame(np.column_stack((mnn_cell1, mnn_cell2)))
    mnns.columns = ['cell1', 'cell2']
    return mnns

#intra = TRUE时，输入都为real-ST，所以输出不需要合并data1和data2，只保留一个即可
def AE_dim_reduction(data1,data2,intra = False):
    encoding_dim = 50
    feature_num = int(np.shape(data1)[0])
    diff = feature_num - encoding_dim
    input_img = Input(shape=(feature_num,))
    #data1 = np.array(data1).astype('int64')
    #data2 = np.array(data2).astype('int64')

    # encoder layers
    encoded = Dense(round(diff/2)+encoding_dim, activation='relu')(input_img)
    encoded = Dense(round(diff/4)+encoding_dim, activation='relu')(encoded)
    encoded = Dense(round(diff/8)+encoding_dim, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(round(diff/8)+encoding_dim, activation='relu')(encoder_output)
    decoded = Dense(round(diff/4)+encoding_dim, activation='relu')(decoded)
    decoded = Dense(round(diff/2)+encoding_dim, activation='relu')(decoded)
    decoded = Dense(feature_num, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)

    # construct the encoder model for plotting
    encoder = Model(input=input_img, output=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    if intra:
        train = data1.transpose()
    else:
        train = pd.concat([data1, data2], axis=1).transpose()
    # training
    autoencoder.fit(train, train,
                    epochs=5,
                    batch_size=200,
                    shuffle=True)

    # output encoded results
    results = pd.DataFrame(encoder.predict(train).transpose())
    #results = pd.DataFrame(train.transpose())
    if intra:
        results.columns = np.array(data1.columns)
    else:
        results.columns = np.concatenate((np.array(data1.columns), np.array(data2.columns)))


    results.index = ['D_' + str(i) for i in range(np.shape(results)[0])]
    results = results.transpose()
    return results

#用embed的方式和直接使用所有feature的方式，分别得到的所有edges，做交集，即为最终filter之后的结果
def filterEdge(edges, neighbors, mats, features, k_filter):
    nn_cells1 = neighbors[4]
    nn_cells2 = neighbors[5]
    mat1 = mats.loc[features, nn_cells1].transpose()
    mat2 = mats.loc[features, nn_cells2].transpose()
    cn_data1 = l2norm(mat1)
    cn_data2 = l2norm(mat2)
    nn = kNN(data=cn_data2.loc[nn_cells2, ],
             query=cn_data1.loc[nn_cells1, ],
             k=k_filter)
    # position = [
    #     np.where(
    #         edges.loc[:, "cell2"][x] == nn[1][edges.loc[:, 'cell1'][x], ])[0]
    #     for x in range(edges.shape[0])
    # ]
    # nps = np.concatenate(position, axis=0)
    # fedge = edges.iloc[nps, ]
    index = []
    for x in range(edges.shape[0]):
        if len(np.where(edges.loc[:, "cell2"][x] == nn[1][edges.loc[:, 'cell1'][x], ])[0]) > 0:
            index.append(x)
    index = np.unique(index)
    fedge = edges.iloc[index,]
    print("\t Finally identified ", fedge.shape[0], " MNN edges")
    return (fedge)

#final_edges前两列为两组细胞，每一行是两个细胞间的一个连线
def Link_graph(count_list,
               norm_list,
               scale_list,
               features,
               combine,
               intra,
               k_filter=200 ):
    all_edges = []
    for row in combine:
        i = row[0]
        j = row[1]
        counts1 = count_list[i]
        counts2 = count_list[j]
        norm_data1 = norm_list[i]
        norm_data2 = norm_list[j]
        scale_data1 = scale_list[i]
        scale_data2 = scale_list[j]
        rowname = counts1.index

        #encoded = AE_dim_reduction(scale_data1,scale_data2,intra)
        cell_embedding, loading = Embed(data_use1=scale_data1,
                                        data_use2=scale_data2,
                                        features=features,
                                        count_names=rowname,
                                        num_cc=3)
        norm_embedding = l2norm(mat=cell_embedding[0])

        #用autoencoder降维方法可代替Embed和l2norm两个function

        cells1 = counts1.columns
        cells2 = counts2.columns
        neighbor = KNN(cell_embedding=norm_embedding,
                       cells1=cells1,
                       cells2=cells2,
                       k=100)
        mnn_edges = MNN(neighbors=neighbor,
                        colnames=cell_embedding[0].index,
                        num=100)
        # select_genes = TopGenes(Loadings=loading,
        #                         dims=range(30),
        #                         DimGenes=100,
        #                         maxGenes=200)
        Mat = pd.concat([norm_data1, norm_data2], axis=1)
        final_edges = filterEdge(edges=mnn_edges,
                                 neighbors=neighbor,
                                 mats=Mat,
                                 features=features,
                                 k_filter=k_filter)

        all_edges.append(final_edges)
    return all_edges
