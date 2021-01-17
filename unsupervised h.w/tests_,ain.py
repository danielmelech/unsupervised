import numpy as np
from scipy import stats
from scipy.stats import f_oneway

def anova (mat):
    anova=f_oneway(mat[0, :], mat[1, :],mat[2, :], mat[3, :],mat[4, :])
    print(anova)
    return None

def anovas (mat):
    anova=f_oneway(mat[0, :], mat[1, :])
    print(anova)
    return None

def kruskal (mat):
    kruskal=stats.kruskal(mat[0, :], mat[1, :],mat[2, :], mat[3, :],mat[4, :])
    print(kruskal)
    return None

def pearson(ss,mat):

    corl_kmeans=stats.pearsonr(ss[0,:], mat[0,:])
    corl_fuzzy=stats.pearsonr(ss[1,:], mat[1,:])
    corl_gmm=stats.pearsonr(ss[2,:], mat[2,:])
    corl_spec=stats.pearsonr(ss[3,:], mat[3,:])
    corl_hirr=stats.pearsonr(ss[4,:], mat[4,:])
    print(corl_kmeans)
    print(corl_fuzzy)
    print(corl_gmm)
    print(corl_spec)
    print(corl_hirr)
    return None

def spearman(ss,mat):
    corl_kmeans = stats.spearmanr(ss[0, :], mat[0, :])
    corl_fuzzy = stats.spearmanr(ss[1, :], mat[1, :])
    corl_gmm = stats.spearmanr(ss[2, :], mat[2, :])
    corl_spec = stats.spearmanr(ss[3, :], mat[3, :])
    corl_hirr = stats.spearmanr(ss[4, :], mat[4, :])
    print(corl_kmeans)
    print(corl_fuzzy)
    print(corl_gmm)
    print(corl_spec)
    print(corl_hirr)
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

if __name__ == '__main__':

    ps_results3= np.genfromtxt("test_rps_main1.csv", delimiter=',')
    silo_results3=np.genfromtxt("test_rs_main1.csv", delimiter=',')
    f_results3=np.genfromtxt("test_rf_main1.csv", delimiter=',')
    ps_results2 = np.genfromtxt("test_rps_main.csv", delimiter=',')
    silo_results2 = np.genfromtxt("test_rs_main.csv", delimiter=',')
    f_results2 = np.genfromtxt("test_rf_main.csv", delimiter=',')

    silo=np.zeros([2,5])
    silo[0,:]=silo_results2
    silo[1, :] = silo_results3

    anova(ps_results2)
    anova(ps_results3)
    anova(f_results2)
    anova(f_results3)
    anovas(silo)

    #np.savetxt("test_rps_diab1.csv", ps_results, fmt='%f',delimiter=',')
    #np.savetxt("test_rf_diab1.csv", f_results, fmt='%f',delimiter=',')
    #np.savetxt("test_rs_diab1.csv", silo_results, fmt='%f',delimiter=',')