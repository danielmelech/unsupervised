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
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


def rand_train(train_x,train_y):

    '''

    :param train_x: matrix of 12D wine points
    :param train_y: labels array
    :return:  matrix of 12D wine points shuffled ,labels array shuffled

    '''

    c = np.concatenate([train_x, train_y.reshape(-1,1)], axis=1)
    np.random.shuffle(c)
    train_y = c[:, -1]
    train_x = c[:, :-1]
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


'''train_x1,train_y1=rand_train(train_x1,train_y1)
    train_x1, train_y1 = divide_t_v(train_x1, train_y1, 20000)'''

def get_data(data):
    train_x1 = np.genfromtxt(data, dtype=str, delimiter=';')
    s = train_x1[:, 7]
    train_y1 = train_x1[:, 4]
    train_x1 = np.delete(train_x1, 4, axis=1)
    train_x1 = np.delete(train_x1, 0, axis=1)
    u = np.unique(s)
    k = np.sort(u)
    for i in range(len(train_x1)):
        train_x1[i][5] = str(np.where(k == train_x1[i][5])[0][0])
    train_x1 = train_x1.astype(float)
    train_y1 = train_y1.astype(float)
    return train_x1,train_y1

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




def kmeans(train_x1,train_y):
    #standarization
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    print(pca.explained_variance_ratio_)
    train_x1 = pca.transform(train_x1)
    kmeans = KMeans(n_clusters=7).fit(train_x1)
    m = kmeans.labels_
    ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    print(ss)
    print(ps)
    a = f1_score(train_y.flatten(), m, average='weighted')
    print(a)
    plot_clusters(train_x1, m)
    return ps,a,ss

def fuuzyc(train_x1,train_y):
    #standarization
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    fcm = FCM(n_clusters=7)
    fcm.fit(train_x1)
    m = fcm.predict(train_x1)
    ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    #print(ss)
    #print(ps)
    a = f1_score(train_y.flatten(), m, average='weighted')
    #print(a)
    #plot_clusters(train_x1, m)
    return ps,a,ss

def gmm(train_x1,train_y):
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    gm = GaussianMixture(n_components=7).fit(train_x1)
    m = gm.predict(train_x1)
    ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    print(ss)
    print(ps)
    a = f1_score(train_y.flatten(), m, average='weighted')
    print(a)
    plot_clusters(train_x1, m)
    return ps,a,ss

def spec(train_x1,train_y):
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    clustering = SpectralClustering(n_clusters=7).fit(train_x1)
    m = clustering.labels_
    ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    #print(ss)
    #print(ps)
    a = f1_score(train_y.flatten(), m, average='weighted')
    #print(a)
    #plot_clusters(train_x1, m)
    return ps,a,ss

def hirr(train_x1,train_y):
    pca = PCA(n_components=2)
    pca.fit(train_x1)
    train_x1 = pca.transform(train_x1)
    clustering = AgglomerativeClustering(n_clusters=7).fit(train_x1)
    m = clustering.labels_
    ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    #print(ss)
    #print(ps)
    a = f1_score(train_y.flatten(), m, average='weighted')
    #print(a)
    #plot_clusters(train_x1, m)
    return ps,a,ss

if __name__ == '__main__':
    data2 = "dataset3.csv"
    train_x,train_y=get_data(data2)
    train_x,train_y=rand_train(train_x,train_y)
    train_x=minmax(train_x)
    train_x3=np.copy(train_x)
    pca = PCA(n_components=2)
    pca.fit(train_x3)
    print(pca.explained_variance_ratio_)
    train_x3 = pca.transform(train_x3)
    plot_clusters(train_x3,train_y)
    kmeans(train_x,train_y)
    gmm(train_x,train_y)

    mg=16500
    rps=np.zeros([5,10])
    rf=np.zeros([5,10])
    rs=np.zeros([5,10])
    for i in range(1):
        print(i)
        cd=np.copy(train_x[i*mg:(i+1)*mg,:])
        cd=minmax(cd)
        cv=np.copy(train_y[i*mg:(i+1)*mg])
        ps1,t1,ss1=kmeans(cd,cv)
        ps2,t2,ss2=fuuzyc(cd,cv)
        ps3,t3,ss3=gmm(cd,cv)
        #ps4,t4,ss4=spec(cd,cv)
        ps5,t5,ss5=hirr(cd,cv)
        rps[0][i] = ps1
        rps[1][i] = ps2
        rps[2][i] = ps3
        #rps[3][i] = ps4
        rps[4][i] = ps5
        rf[0][i] = t1
        rf[1][i] = t2
        rf[2][i] = t3
        #rf[3][i] = t4
        rf[4][i] = t5
        rs[0][i] = ss1
        rs[1][i] = ss2
        rs[2][i] = ss3
        #rs[3][i] = ss4
        rs[4][i] = ss5
    print(rps)
    print(rf)
    print(rs)
    np.savetxt("test_rps", rps, fmt='%f')
    np.savetxt("test_rf", rf, fmt='%f')
    np.savetxt("test_rs", rs, fmt='%f')



