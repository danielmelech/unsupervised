import numpy as np
from scipy import stats
from scipy.stats import f_oneway

def tests (mat):
    ttest = stats.ttest_ind(mat[0, :], mat[1, :])
    anova=f_oneway(mat[0, :], mat[1, :],mat[2, :], mat[3, :],mat[4, :])
    print(anova)
    return None

def anova (mat):
    anova=f_oneway(mat[0, :], mat[1, :],mat[2, :], mat[3, :],mat[4, :])
    print(anova)
    return None

def kruskal (mat):
    kruskal=stats.kruskal(mat[0, :], mat[1, :],mat[2, :], mat[3, :],mat[4, :])
    print(kruskal)
    return None

def shapiro(mat):
    corl_kmeans = stats.shapiro( mat[0, :])
    corl_fuzzy = stats.shapiro( mat[1, :])
    corl_gmm = stats.shapiro( mat[2, :])
    corl_spec = stats.shapiro( mat[3, :])
    corl_hirr = stats.shapiro( mat[4, :])
    print(corl_kmeans)
    print(corl_fuzzy)
    print(corl_gmm)
    print(corl_spec)
    print(corl_hirr)
    return None

if __name__ == '__main__':
    data="results_country"
    print(f_oneway([1,2],[1,2]))
    train_x1 = np.genfromtxt(data, delimiter='')
    ps=train_x1[0:5,:]
    rsf=train_x1[5:10,:]
    ss=train_x1[10:15,:]
    anova(ps)
    anova(rsf)
    kruskal(ss)
    '''kruskal(ps)
    kruskal(rsf)
    kruskal(ss)'''

    '''tests(ps)
    tests(rsf)
    tests(ss)
    kruskal(ps)
    kruskal(rsf)
    kruskal(ss)'''
    #np.savetxt("country.csv", train_x1, fmt='%f',delimiter=',')
