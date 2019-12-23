# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 01:17:50 2019

@author: dimit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nr
from sklearn import metrics
from scipy.stats import mode
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.decomposition as skde
from sklearn.cluster import KMeans, AgglomerativeClustering
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
""" 
Cleaning the dataset
:param brcancer: the dataset
:return: the cleaned dataset
"""
def clean_auto_data(brcancer):
    cols = brcancer.columns
    brcancer.columns = [str.replace(' ', '_') for str in cols]
    cols = brcancer.columns[2:]
    for column in cols:
        brcancer[column] = pd.to_numeric(brcancer[column])
    return brcancer

""" 
Plots the density histogram for all features
:param brcancer: the dataset
:param cols: the feature's names
:param bins: the number of bins
:param hist: add the histogram to the plot
:param name: the name of the file
"""
def plot_density_hist(brcancer, cols, bins, name):
    for col in cols:
        fig = plt.figure(figsize=(6, 6))
        plt.style.use('seaborn-darkgrid')
        sns.distplot(brcancer[col], bins=bins, rug=True, hist=True)
        plt.title('Histogram of ' + col)
        plt.xlabel(col)
        plt.ylabel('Number of patients')
        plt.show()
        fig.savefig(name + col + '.png')

""" 
Plots the explained variance for every number of components in PCA
:param mod: the PCA model
"""       
def plot_explained_variance(mod):
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    x = [y + 1 for y in x] 
    plt.style.use('seaborn-darkgrid')         
    plt.plot(x,comps)    
    plt.title('Explained variance vs number of components')
    plt.xlabel('Number of components for PCA')
    plt.ylabel('Explained variance')
    plt.show()
    
""" 
Return a vector of the most important features from the PCA components
:param comps: the PCA components
:param most_important: the index of the most important features
:return: the most important features
"""
def most_important_features(comps,most_important):
    features = []
    for j in most_important:
        features.append(abs(comps[j]))
    return features

""" 
Mapping the predicted labels to the true labels
:param assign: the predicted labels
:param brcancer_indx: the data used for the prediction
:return: the mapped true labels
"""
def create_labels(assign,brcancer_indx):
     labels = np.zeros_like(assign)
     for i in range(2):
         mask = (assign == i)
         digits_mask = brcancer_indx[mask]
         labels[mask] = mode(digits_mask)[0]
     return labels
 
""" 
Calculates the mean of performance scores after cross validation
:param scores: the dictionary of the scores for a performance measure
:return: the mean of the performance measure
"""
def calculate_mean(scores):
    means = []
    for i in range(0,5):
        mean = np.mean(scores[:, i])
        means.append(mean)
    return means

""" 
Plots the following:
    - the heat map of the contribution of each feature for different PCA components
    - the barplot of the contrubution of the 3 most important features
    (each from the first and the second principal components) for different PCA components
:param pca_mod: the estimated PCA model on the training data
"""
def initial_plots(pca_mod):
    # number of components
    n_pcs = pca_mod.components_.shape[0]
    
    # PCA heat plot
    plt.matshow(pca_mod.components_,cmap='coolwarm')
    plt.yticks([0,1,2,3,4],['1st Comp','2nd Comp','3rd Comp', '4th comp', '5th comp'],fontsize=9)
    plt.colorbar()
    plt.xticks(range(len(cols)),cols,rotation=65,ha='left',fontsize=9)
    plt.tight_layout()
    plt.show()
    
    most_important = [5, 13, 6, 9, 19, 14]
    
    pca_comp_values = {}
    for i in range(n_pcs):
        comps = pca_mod.components_[i]
        idx = 'PCA'+ str(i);
        pca_comp_values[idx] = most_important_features(comps,most_important)
    
    # PCA plot
    pos = list(range(len(pca_comp_values['PCA0'])))
    width = 0.1
    palette = sns.color_palette('Spectral',10)
    fig, ax = plt.subplots(figsize=(10, 5))

    plt.bar(pos,pca_comp_values['PCA0'],width,alpha=0.5,color=palette[0],
            label=cols[most_important[0]])
    plt.bar([p + width for p in pos],pca_comp_values['PCA1'],width,alpha=0.5,color=palette[1],
            label=cols[most_important[1]])
    plt.bar([p + width * 2 for p in pos],pca_comp_values['PCA2'],width,alpha=0.5,
            color=palette[2],label=cols[most_important[2]])
    plt.bar([p + width * 3 for p in pos],pca_comp_values['PCA3'],width,alpha=0.5,
            color=palette[7],label=cols[most_important[3]])
    plt.bar([p + width * 4 for p in pos],pca_comp_values['PCA4'],width,alpha=0.5,
            color=palette[9],label=cols[most_important[4]])

    ax.set_ylabel('Importance Score')
    ax.set_title('Features')
    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(cols[most_important[i]] for i in range(0, len(most_important)))

    plt.xlim(min(pos) - width, max(pos) + width * 5)
    plt.ylim([0, max(pca_comp_values['PCA0'] + pca_comp_values['PCA1'] + pca_comp_values['PCA2'] + pca_comp_values['PCA3'] + pca_comp_values['PCA4'])])

    plt.legend(['PCA0', 'PCA1', 'PCA2', 'PCA3', 'PCA4'], loc='upper left')
    plt.grid()
    plt.show()

""" 
Plots the following:
    - the scatter plot of the two categories (B and M), using the predicted and the true labels
:param Comps: the principal components of the training set
:param Comps_test: the principal components of the test set (applied only for kmeans algorithm)
:param indx: the indices of the sample used for training
:param algo: the clustering algorithm
:return: the performance scores (accuracy,recall,precision,F1)
"""
def cluster_plot(Comps,Comps_test,indx,algo):    
    if algo=='kmeans':
        kmeans= KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
        assignments = kmeans.fit_predict(Comps)
        assignments_test= kmeans.predict(Comps_test)
    elif algo=='hierarchical':
        agg= AgglomerativeClustering(n_clusters=2)
        assignments= agg.fit_predict(Comps)
    
     #Training
    labels_training = create_labels(assignments, brcancer['Diagnosis'][indx[0]])
    print(labels_training)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    ax1.scatter(Comps[:,0],Comps[:,1],c=labels_training , cmap = "jet", edgecolor = "None", alpha=0.35)
    ax1.set_title(algo + ' clustering plot')
    
    ax2.scatter(Comps[:,0],Comps[:,1],c = brcancer['Diagnosis'][indx[0]], cmap = "jet", edgecolor = "None", alpha=0.35)
    ax2.set_title('Actual clusters')
    plt.show()
    
    if algo=='kmeans':
        labels_test = create_labels(assignments_test, brcancer['Diagnosis'][indx[1]])
        print(labels_test)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        ax1.scatter(Comps_test[:,0],Comps_test[:,1],c=labels_test , cmap = "jet", edgecolor = "None", alpha=0.35)
        ax1.set_title('k-means clustering plot')
        
        ax2.scatter(Comps_test[:,0],Comps_test[:,1],c = brcancer['Diagnosis'][indx[1]], cmap = "jet", edgecolor = "None", alpha=0.35)
        ax2.set_title('Actual clusters')
        plt.show()
    
    performance_scores = {}
    labels_true = brcancer['Diagnosis'][indx[0]]    
    labels_pred = assignments

    ACCURACY=accuracy_score(labels_true, labels_pred)
    RECALL=recall_score(labels_true, labels_pred)
    PRECISION=precision_score(labels_true, labels_pred)
    F1=f1_score(labels_true, labels_pred,'binary')
    performance_scores['Accuracy'] = ACCURACY
    performance_scores['Recall'] = RECALL
    performance_scores['Precision'] = PRECISION
    performance_scores['F1'] = F1
    
    return(performance_scores)
""" 
The initial run using 2 clusters for each clustering algorithm (kmeans and hierarchical)
:param n_comps: the number of components for PCA
:param test_sz: the size of the test set
:return: the performance scores of the two clustering algorithms (accuracy,recall,precision,F1)
"""        
def initial_run(n_comps,test_sz):
    nr.seed(9988)
    Features = np.array(brcancer_sc)
    indx = range(Features.shape[0])
    indx = ms.train_test_split(indx, test_size=test_sz)
    
    x_train = Features[indx[0],:]
    x_test = Features[indx[1],:]
    
    # Rescale numeric features
    scaler = preprocessing.StandardScaler().fit(x_train[:,:])
    x_train[:, :] = scaler.transform(x_train[:,:])
    x_test[:, :] = scaler.transform(x_test[:, :])
    
    print(x_train.shape)
    #plot_density_hist(x_train,cols, bins=20, hist=True, name='scaled_dens_')
    
    pca_mod = skde.PCA()
    pca_comps = pca_mod.fit(x_train)
    pca_comps
    print(pca_comps.explained_variance_ratio_)
    print(np.sum(pca_comps.explained_variance_ratio_))
    
    plot_explained_variance(pca_comps)
    pca_mod = skde.PCA(n_components = n_comps)
    pca_mod.fit(x_train)
    Comps = pca_mod.transform(x_train)
    Comps.shape
    Comps_test=pca_mod.transform(x_test)
    initial_plots(pca_mod)

    cluster_plot(Comps,Comps_test,indx,'kmeans') 
    cluster_plot(Comps,Comps_test,indx,'hierarchical') 
    return(cluster_plot(Comps,Comps_test,indx,'kmeans'), cluster_plot(Comps,Comps_test,indx,'hierarchical') )

""" 
Performs cross validation on the dataset
:param n_folds: the number of folds used for cross validation
:param n_clusters: the array containing the number of clusters
:param data: the initial dataset
:param data_transformed: the scaled and transformed features of the dataset
:param algo: the clustering algorithm
:return: the dictionary of different performance scores for the clustering algorithm used
"""
def cross_valid(n_folds,n_clusters,data,data_transformed,algo):
    performance_scores = {}
    nk = len(n_clusters)
    random_order=np.arange(0,brcancer_sc.shape[0]+1,1)
    random.seed(120)
    random.shuffle(random_order)
    lims=np.arange(0,random_order[0]-1,int(round(brcancer_sc.shape[0]/n_folds)))
    lims=np.append(lims,random_order[0])
    print(lims)
    u=1
    ARI=np.zeros(shape=(n_folds, nk))
    AMI=np.zeros(shape=(n_folds,nk))
    NMI=np.zeros(shape=(n_folds,nk))
    H=np.zeros(shape=(n_folds,nk))
    C=np.zeros(shape=(n_folds,nk))
    VHC=np.zeros(shape=(n_folds,nk))
    FMS=np.zeros(shape=(n_folds,nk))
    CHS=np.zeros(shape=(n_folds,nk))
    SIL=np.zeros(shape=(n_folds,nk))
    for k in range(len(lims)-1):
        n = len(lims) - 1
        Features = np.array(data_transformed)
        x_train = Features[lims[0]:lims[n],:]
        x_test = Features[lims[n-u]:lims[n-u+1],:]
        train_set = np.arange(lims[0], lims[n], 1)
        test_set=np.arange(lims[n-u],lims[n-u+1],1)
        train_set = np.delete(train_set, test_set, 0)
        x_train=np.delete(x_train,test_set,0)
        u=u+1
        
        # Rescale numeric features,
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train= scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        pca_mod_5 = skde.PCA(n_components = 5)
        pca_mod_5.fit(x_train)
        Comps= pca_mod_5.transform(x_train)
    
        for i in n_clusters:
            if algo=='kmeans':
                kmeans= KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                    precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
                assignments = kmeans.fit_predict(Comps)
            elif algo=='hierarchical':
                agg= AgglomerativeClustering(n_clusters=i)
                assignments= agg.fit_predict(Comps)
    
            labels_true = data['Diagnosis'][train_set]
            labels_pred = assignments
            ARI[k,i-2]=metrics.adjusted_rand_score(labels_true, labels_pred)
            AMI[k,i-2]=metrics.adjusted_mutual_info_score(labels_true, labels_pred)
            NMI[k,i-2]=metrics.normalized_mutual_info_score(labels_true, labels_pred)
            H[k,i-2]=metrics.homogeneity_score(labels_true, labels_pred)
            C[k,i-2]=metrics.completeness_score(labels_true, labels_pred)
            VHC[k,i-2]=metrics.v_measure_score(labels_true, labels_pred)
            FMS[k,i-2]=metrics.fowlkes_mallows_score(labels_true, labels_pred)
            CHS[k,i-2]=metrics.calinski_harabasz_score(Comps, labels_pred)
            SIL[k,i-2]=metrics.silhouette_score(Comps[:,0:1],labels_pred)
            
    performance_scores['ARI'] = calculate_mean(ARI)
    performance_scores['AMI'] = calculate_mean(AMI)
    performance_scores['NMI'] = calculate_mean(NMI)
    performance_scores['H'] = calculate_mean(H)
    performance_scores['C'] = calculate_mean(C)
    performance_scores['VHC'] = calculate_mean(VHC)
    performance_scores['FMS'] = calculate_mean(FMS)
    performance_scores['CHS'] = calculate_mean(CHS)
    performance_scores['SIL'] = calculate_mean(SIL)
    return(performance_scores)
    
""" 
Plots the performance score of two clustering algorithm for different number of clusters
:param k_val: the array with the different number of clusters
:param performance_scores1: the performance scores of the first algorithm
:param performance_scores2: the performance scores of the second algorithm
""" 
def plot_scores(k_val,performance_scores1,performance_scores2):
    df1=pd.DataFrame({'x': k_val, 'SIL': performance_scores1['SIL'],
                     'FMS': performance_scores1['FMS'],
                     'ARI': performance_scores1['ARI'],
                     'NMI': performance_scores1['NMI']})
    df2=pd.DataFrame({'x': k_val, 'SIL': performance_scores2['SIL'],
                     'FMS': performance_scores2['FMS'],
                     'ARI': performance_scores2['ARI'],
                     'NMI': performance_scores2['NMI']})
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    num=0
    for column in df1.drop('x', axis=1):
        num+=1
        km, =ax.plot(df1['x'], df1[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
        hier, =ax.plot(df2['x'], df2[column], linestyle='-.', color=palette(num), linewidth=1, alpha=0.9)
    leg1=ax.legend(loc = 3)
    leg2=ax.legend([km,hier],['kmeans','hierarchical'],loc=1)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.set_title('Average Performance scores after cross validation')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Performance Score')
    plt.show()
    

#Load dataset and encode the labels
brcancer = pd.read_csv('data.csv')
brcancer['Diagnosis'] = brcancer['Diagnosis'].map({'M':1,'B':0})    
brcancer = clean_auto_data(brcancer)
#Define the numerical columns
cols=brcancer.columns[2:]

#Scatter plot of all the numerical features, in order to observe the relationships
#sns.pairplot(brcancer[cols], palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")

#Log trasformation of the dataset in order to minimize the skewness
brcancer_sc = pd.DataFrame()
for col in cols:
    brcancer_sc[col] = brcancer[col]
    if (col != 'mean_symmetry' and col != 'mean_smoothness' and col != 'worst_smoothness'):
        brcancer_sc[col] = np.log(brcancer[col])

brcancer_sc.replace([np.inf, -np.inf], 0, inplace=True)

#Plot the histogram to examine the changes after transformation
#plot_density_hist(brcancer,cols, bins = 20, name='density_')
#plot_density_hist(brcancer_sc,cols, bins = 20, name='density_')

#Perform the initial run to analyze two clusters
scores_km,scores_agg=initial_run(5,114)

#Cross validation using different number of clusters for both k-means and hierarchical algorithm
k_val=[2,3,4,5,6]
nf=5
performance_scores_km=cross_valid(nf,k_val,brcancer,brcancer_sc,'kmeans')
performance_scores_agg =cross_valid(nf,k_val,brcancer,brcancer_sc,'hierarchical') 

#Comparison of clustering algorithms based on the performance scores
plot_scores(k_val,performance_scores_km,performance_scores_agg)

