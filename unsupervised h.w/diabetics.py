import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import v_measure_score
from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import homogeneity_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from fcmeans import FCM
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from scipy.stats import entropy
from scipy import stats
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.cm as cm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering



def rand_train(train_x,train_y):

    '''

    :param train_x: matrix of 12D wine points
    :param train_y: labels array
    :return:  matrix of 12D wine points shuffled ,labels array shuffled

    '''

    c = np.concatenate([train_x, train_y], axis=1)
    np.random.shuffle(c)
    train_y = c[:, -4:-1]
    train_x = c[:, :-3]
    return train_x,train_y


def divide_t_l(train_x,t):


    train_x1=train_x[:,:t]
    labels=train_x[:,t:]
    labels = labels.astype(int)
    return train_x1,labels

def divide_t_v(train_x,train_y,t):


    train_x1=train_x[:t,:]
    train_y1=train_y[:t]
    #valid_x=train_x[t:,:]
    #valid_y=train_y[t:]
    train_y1 = train_y1.astype(int)
    #valid_y=valid_y.astype(int)
    return train_x1,train_y1

def get_data(data):
    train_x1 = np.genfromtxt(data, dtype=float, delimiter=',')
    tag = train_x1[:, 0:3]
    #train_x1,tag=rand_train(train_x1,tag)
    #train_x1, tag=divide_t_v(train_x1,tag,20000)
    gender=tag[:,1]
    race=tag[:,0]
    train_x1 = np.delete(train_x1, 2, axis=1)
    train_x1 = np.delete(train_x1, 1, axis=1)
    train_x1 = np.delete(train_x1, 0, axis=1)
    tag = np.delete(tag, 1, axis=1)
    tag = np.delete(tag, 0, axis=1)
    train_x1 = train_x1.astype(float)
    tag = tag.astype(int)
    gender = gender.astype(int)
    race = race.astype(int)
    return train_x1,tag,gender,race

def minmax(train_x):
    scaler = MinMaxScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    return train_x

def standarization(train_x):
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    return train_x

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def plot_clusters(train_x1,labels):
    u_labels = np.unique(labels)

    # plotting the results:

    for i in u_labels:
        plt.scatter(train_x1[labels == i, 0], train_x1[labels == i, 1], label=i)
    plt.legend()
    plt.show()
    return None

def kmeans(train_x1,tag,gender,race):
    #standarization
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    kmeans = KMeans(n_clusters=9).fit(train_x1)
    m = kmeans.labels_
    ps=[]
    ps.append(purity_score(tag, m))
    ps.append(purity_score(gender, m))
    ps.append(purity_score(race, m))
    ss = silhouette_score(train_x1, m)
    print(ss)
    print(ps)
    a=[]
    a.append(f1_score(tag.flatten(), m, average='weighted'))
    a.append(f1_score(gender.flatten(), m, average='weighted'))
    a.append(f1_score(race.flatten(), m, average='weighted'))
    print(a)
    plot_clusters(train_x1, m)
    return ps,a,ss

def fuuzyc(train_x1,tag,gender,race):
    #standarization
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    fcm = FCM(n_clusters=9)
    fcm.fit(train_x1)
    m = fcm.predict(train_x1)
    ps = []
    ps.append(purity_score(tag, m))
    ps.append(purity_score(gender, m))
    ps.append(purity_score(race, m))
    ss = silhouette_score(train_x1, m)
    #print(ss)
    #print(ps)
    a = []
    a.append(f1_score(tag.flatten(), m, average='weighted'))
    a.append(f1_score(gender.flatten(), m, average='weighted'))
    a.append(f1_score(race.flatten(), m, average='weighted'))
    #print(a)
    # plot_clusters(train_x1, m)
    return ps,a,ss

def gmm(train_x1,tag,gender,race):
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    gm = GaussianMixture(n_components=9).fit(train_x1)
    m = gm.predict(train_x1)
    ps = []
    ps.append(purity_score(tag, m))
    ps.append(purity_score(gender, m))
    ps.append(purity_score(race, m))
    ss = silhouette_score(train_x1, m)
    print(ss)
    print(ps)
    a = []
    a.append(f1_score(tag.flatten(), m, average='weighted'))
    a.append(f1_score(gender.flatten(), m, average='weighted'))
    a.append(f1_score(race.flatten(), m, average='weighted'))
    print(a)
    plot_clusters(train_x1, m)
    return ps,a,ss

def spec(train_x1,tag,gender,race):
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    clustering = SpectralClustering(n_clusters=9).fit(train_x1)
    m = clustering.labels_
    ps = []
    ps.append(purity_score(tag, m))
    ps.append(purity_score(gender, m))
    ps.append(purity_score(race, m))
    ss = silhouette_score(train_x1, m)
    # print(ss)
    # print(ps)
    a = []
    a.append(f1_score(tag.flatten(), m, average='weighted'))
    a.append(f1_score(gender.flatten(), m, average='weighted'))
    a.append(f1_score(race.flatten(), m, average='weighted'))
    # print(a)
    # plot_clusters(train_x1, m)
    return ps,a,ss

def hirr(train_x1,tag,gender,race):
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    clustering = AgglomerativeClustering(n_clusters=9).fit(train_x1)
    m = clustering.labels_
    ps = []
    ps.append(purity_score(tag, m))
    ps.append(purity_score(gender, m))
    ps.append(purity_score(race, m))
    ss = silhouette_score(train_x1, m)
    print(ss)
    print(ps)
    a = []
    a.append(f1_score(tag.flatten(), m, average='weighted'))
    a.append(f1_score(gender.flatten(), m, average='weighted'))
    a.append(f1_score(race.flatten(), m, average='weighted'))
    print(a)
    plot_clusters(train_x1, m)
    return ps,a,ss


if __name__ == '__main__':
    data2 = "dataset2.csv"
    train_x,tag,gender,race=get_data(data2)
    mg = 20000
    rps = np.zeros([15, 10])
    rf = np.zeros([15, 10])
    rs = np.zeros([5, 10])
    for i in range(10):
        print(i)
        cd = np.copy(train_x[i * mg:(i + 1) * mg, :])
        cvt = np.copy(tag[i * mg:(i + 1) * mg])
        cvg = np.copy(gender[i * mg:(i + 1) * mg])
        cvr = np.copy(race[i * mg:(i + 1) * mg])
        cd = minmax(cd)
        ps1, t1, ss1 = kmeans(cd, cvt,cvg,cvr)
        ps2, t2, ss2 = fuuzyc(cd, cvt,cvg,cvr)
        ps3, t3, ss3 = gmm(cd,cvt,cvg,cvr)
        ps4,t4,ss4=spec(cd,cvt,cvg,cvr)
        ps5, t5, ss5 = hirr(cd,cvt,cvg,cvr)
        for j in range(3):
            rps[0+5*j][i] = ps1[j]
            rps[1+5*j][i] = ps2[j]
            rps[2+5*j][i] = ps3[j]
            rps[3+5*j][i] = ps4[j]
            rps[4+5*j][i] = ps5[j]
            rf[0+5*j][i] = t1[j]
            rf[1+5*j][i] = t2[j]
            rf[2+5*j][i] = t3[j]
            rf[3+5*j][i] = t4[j]
            rf[4+5*j][i] = t5[j]
        rs[0][i] = ss1
        rs[1][i] = ss2
        rs[2][i] = ss3
        rs[3][i] = ss4
        rs[4][i] = ss5
    print(rps)
    print(rf)
    print(rs)
    np.savetxt("test_rps_diab", rps, fmt='%f')
    np.savetxt("test_rf_diab", rf, fmt='%f')
    np.savetxt("test_rs_diab", rs, fmt='%f')
