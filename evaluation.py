import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import ndcg_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch


def tSNE(data, labels, cmap="", func=None, ptsize=35, init="pca", lr=200.0, filename=""):
    tsne = TSNE(2, init=init)
    tsne.learning_rate = lr
    tsne_result = tsne.fit_transform(data)
    tsne_result_df = pd.DataFrame({'x': tsne_result[:,0], 'y': tsne_result[:,1], 'label':np.array(labels[:])})
    fig, ax = plt.subplots(1)
    if cmap != "":
        sns.scatterplot(x='x', y='y', hue='label', data=tsne_result_df, ax=ax, s=ptsize, picker=True, palette=cmap)
    else:
        sns.scatterplot(x='x', y='y', hue='label', data=tsne_result_df, ax=ax, s=ptsize, picker=True)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    if func is not None:
        fig.canvas.mpl_connect('pick_event', func)
    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    if filename != "":
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return tsne_result[:, 0], tsne_result[:, 1]


def triplet_semantic_correspondence(s1, o1, s2, o2):
    value = 0.0
    if s1 == s2 and o1 == o2:
        value += 1.0
    elif s1 == o2 and o1 == s2:
        value += 1.0
    elif s1 == s2 or s1 == o2:
        value += .5
    elif o1 == s2 or o1 == o2:
        value += .5
    return value


def spatial_classes(fbanners, clusters=4):
    if clusters == 0:
        return cosine_similarity(fbanners, Y=None)
    else:
        pca = PCA(n_components=2)
        x = pca.fit_transform(fbanners)
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(x)
        # plt.scatter(x[:, 0], x[:, 1], c=spatial_labels)
        # plt.show()
        return kmeans.labels_


def CBIR(n, outputs, relations, subj, obj, spatial_data, dcg_mat, weights):
    NDCG = 0
    attributes = [0, 0, 0]
    n_images = len(outputs)
    similarities = cosine_similarity(outputs, Y=None)

    for i in range(n_images):
        image_value = []
        idx_sort = np.argsort(-similarities[i])
        rel = relations[i]
        o1 = subj[i]
        o2 = obj[i]
        sp = spatial_data[i]
        corrs = [0, 0, 0]
        for j in range(1, n + 1):
            idx = idx_sort[j]   
            p_rel = relations[idx]
            p_o1 = subj[idx]
            p_o2 = obj[idx]
            p_sp = spatial_data[idx]
            name_corr = triplet_semantic_correspondence(o1, o2, p_o1, p_o2)
            rel_corr = 1 if rel == p_rel else 0
            sp_corr = spatial_distance(s1=i, s2=j, mat=spatial_data)
            corr = [name_corr, rel_corr, sp_corr]
            corrs[0] += name_corr
            corrs[1] += rel_corr
            corrs[2] += sp_corr
            image_value.append(corr[0] * weights[0] + corr[1] * weights[1] + corr[2] * weights[2])
        discounted = dcg_v(image_value)
        ideal = idcg_v(dcg_mat[i], n)
        if ideal == 0:
            normalized = 0
        else:
            # print(discounted, ideal)
            normalized = discounted / ideal
        NDCG += normalized
        attributes[0] += corrs[0]
        attributes[1] += corrs[1]
        attributes[2] += corrs[2]
    NDCG /= n_images
    percentage = [attributes[k] / (n_images * n) for k in range(3)]
    return NDCG, percentage


def dcg_matrix_computation(relations, subj, obj, spatial_data, weights):
    n_images = len(relations)
    dcg_mat = np.zeros((n_images, n_images))
    for i in range(n_images):
        rel = relations[i]
        o1 = subj[i]
        o2 = obj[i]
        sp = spatial_data[i]
        for j in range(n_images):
            if i != j:
                p_rel = relations[j]
                p_o1 = subj[j]
                p_o2 = obj[j]
                p_sp = spatial_data[j]
                name_corr = triplet_semantic_correspondence(o1, o2, p_o1, p_o2)
                rel_corr = 1 if rel == p_rel else 0
                sp_corr = spatial_distance(s1=i, s2=j, mat=spatial_data)
                val = name_corr * weights[0] + rel_corr * weights[1] + sp_corr * weights[2]
                dcg_mat[i, j] = val
    return dcg_mat


def dcg_p(values, p):
    denom = math.log2(p+1)
    return values / denom

def dcg_v(values_list):
    value = 0
    for i in range(len(values_list)):
        p = i+1
        value += dcg_p(values_list[i], p)
    return value

def idcg_v(matrix, p):
    value = 0
    ideal = np.sort(matrix)
    for i in range(1, p+1):
        value += dcg_p(ideal[-(i)], i)
    return value


def spatial_distance(s1=0, s2=0, mat=None):
    if mat is None:
        if s1 == s2:
            return 1
        return 0
    else:
        return 1-mat[s1, s2]
    

def spatial_distance_matrix(data):
    n_images = len(data)
    dist_mat = np.zeros((n_images, n_images))
    dist_mat = cosine_similarity(data, Y=None)
    return (dist_mat - np.min(dist_mat)) / (np.max(dist_mat) - np.min(dist_mat))



def Precision_spatial(n, outputs, spatial_data):
    prec = []
    matrix = cosine_similarity(outputs, Y=None)
    for i in range(len(outputs)):
        el = spatial_data[i]
        indices = np.argsort(matrix[i])
        predictions = []
        for j in range(1, n + 1):
            val_pred = 1 if spatial_data[indices[-(j+1)]] == el else 0
            predictions.append(val_pred)
        prec.append(sum(predictions)/n)
    return np.mean(prec)

def NDCG_spatial_classes(n, outputs, spatial_data):
    NDCG = []
    matrix = cosine_similarity(outputs, Y=None)
    for i in range(len(outputs)):
        idcg = 0
        dcg = 0
        el = spatial_data[i]
        indices = np.argsort(matrix[i])
        ground_truth = []
        predictions = []
        for j in range(1, n + 1):
            val_gt = 1
            val_pred = 1 if spatial_data[indices[-(j+1)]] == el else 0
            ground_truth.append(val_gt)
            predictions.append(val_pred)
        for k in range(len(ground_truth)):
            log = math.log2((k+1) + 1)
            dcg += predictions[k] / log
            idcg += ground_truth[k] / log
        NDCG.append(dcg/idcg)
    return np.mean(NDCG)


def NDCG_spatial(n, outputs, spatial_data):
    NDCG = []
    matrix = 1 - (cosine_distances(outputs, Y=None) / 2)
    spatial_matrix = spatial_data
    for i in range(len(outputs)):
        idcg = 0
        dcg = 0
        indices = np.argsort(matrix[i]) # model's ranking
        spatial_indices = np.argsort(spatial_matrix[i]) # ground truth ranking
        ground_truth = []
        predictions = []
        for j in range(1, n + 1):
            val_gt = spatial_matrix[i][spatial_indices[-(j+1)]]
            val_pred = spatial_matrix[i][indices[-(j+1)]]
            ground_truth.append(val_gt)
            predictions.append(val_pred)
        for k in range(len(ground_truth)):
            log = math.log2((k+1) + 1)
            dcg += predictions[k] / log
            idcg += ground_truth[k] / log
        NDCG.append(dcg/idcg)
    return np.mean(NDCG)


def NDCG_labels(n, outputs, relations):
    NDCG = []
    matrix = cosine_similarity(outputs, Y=None)
    for i in range(len(outputs)):
        idcg = 0
        dcg = 0
        indices = np.argsort(matrix[i])
        rel = relations[i]
        ground_truth = []
        predictions = []
        for j in range(1, n + 1):
            ground_truth.append(1)
            predictions.append(1 if rel == relations[indices[-(j+1)]] else 0)
        for k in range(len(ground_truth)):
            log = math.log2((k+1)+1)
            dcg += predictions[k] / log
            idcg += ground_truth[k] / log
        NDCG.append(dcg/idcg)
    return np.mean(NDCG)



def onehot(k, n):
    encoding = np.zeros((n,), dtype=np.float32)
    encoding[k] = 1.0
    return encoding

def num_true_positives(logits, labels):
    x = torch.argmax(logits, dim=1)
    y = torch.argmax(labels, dim=1)
    pos = torch.sum(torch.eq(x, y))
    return pos


def acc_at_k(logits, labels, k=3):
    x_k = torch.topk(logits, k, dim=1).indices
    x_kt = x_k.T
    y = torch.argmax(labels, dim=1)
    at1 = torch.sum(torch.eq(x_kt[0], y))
    at2 = torch.sum(torch.eq(x_kt[1], y))
    at3 = torch.sum(torch.eq(x_kt[2], y))
    at4 = torch.sum(torch.eq(x_kt[3], y))
    at5 = torch.sum(torch.eq(x_kt[4], y))
    return at1.item(), at2.item(), at3.item(), at4.item(), at5.item()