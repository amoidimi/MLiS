# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 01:17:50 2019

@author: dimit
"""

# Visualising distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
import numpy.random as nr
import math

from scipy.stats import mode
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import sklearn.decomposition as skde
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mpl_toolkits import mplot3d

brcancer = pd.read_csv('data.csv')
cols = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'std_radius', 'std_texture', 'std_perimeter', 'std_area', 'std_smoothness',
        'std_compactness', 'std_concavity', 'std_concave_points', 'std_symmetry', 'std_fractal_dimension',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
brcancer['Diagnosis'] = brcancer['Diagnosis'].map({'M':1,'B':0})

def clean_auto_data(brcancer):
    cols = brcancer.columns
    brcancer.columns = [str.replace(' ', '_') for str in cols]
    ## Transform column data type
    ## Convert some columns to numeric values
    cols = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
            'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
            'std_radius', 'std_texture', 'std_perimeter', 'std_area', 'std_smoothness',
            'std_compactness', 'std_concavity', 'std_concave_points', 'std_symmetry', 'std_fractal_dimension',
            'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
            'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
    for column in cols:
        brcancer.loc[brcancer[column] == '?', column] = np.nan
    brcancer.dropna(axis=0, inplace=True)
    for column in cols:
        brcancer[column] = pd.to_numeric(brcancer[column])

    return brcancer

brcancer = clean_auto_data(brcancer)
print(brcancer.columns)
brcancer.head()
brcancer.describe()
brcancer.info()
#sns.pairplot(brcancer[cols], palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")

#sns.pairplot(brcancer[cols], palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")
def plot_histogram(brcancer, cols, bins):
    for col in cols:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        brcancer[col].plot.hist(ax=ax, bins=bins)
        ax.set_title('Histogram of ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel('Number of patients')
        plt.show()



plot_histogram(brcancer, cols, 10)


def plot_density_hist(brcancer, cols, bins, hist, name):
    for i in range(0, len(cols)):
        col = cols[i]
        fig = plt.figure(figsize=(6, 6))
        sns.set_style("whitegrid")
        sns.distplot(brcancer[i], bins=bins, rug=True, hist=hist)
        plt.title('Histogram of ' + col)
        plt.xlabel(col)
        plt.ylabel('Number of patients')
        plt.show()
        fig.savefig(name + col + '.png')
        
plot_density_hist(brcancer,cols, bins = 20, hist = True,name='density_')


def plot_scatter(brcancer, cols, col_y='mean_radius'):
    for col in cols:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.gca()
        brcancer.plot.scatter(x=col, y=col_y, ax=ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()
        fig.savefig('scatterplot_' + col + '.png')


plot_scatter(brcancer,cols)

plot_scatter(brcancer, ['mean_texture'], 'mean_perimeter')


def plot_scatter_t(brcancer, cols, col_y='mean_radius', alpha=1.0):
    for col in cols:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.gca()  # define axis
        brcancer.plot.scatter(x=col, y=col_y, ax=ax, alpha=alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()
    # fig.savefig('scatterplot.png')


plot_scatter_t(brcancer,cols, alpha=0.2)


def plot_desity_2d(brcancer, cols, col_y='mean_radius', kind='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=brcancer, kind=kind)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

plot_desity_2d(brcancer,cols)
plot_desity_2d(brcancer,cols, kind='hex')


def plot_scatter_shape(brcancer, cols, shape_col='Diagnosis', col_y='mean_radius', alpha=0.2):
    shapes = ['o', 'x']
    unique_cats = brcancer[shape_col].unique()
    for col in cols:
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats):
            temp = brcancer[brcancer[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker=shapes[i], label=cat,
                        scatter_kws={"alpha": alpha}, fit_reg=False, color='blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.legend()
        plt.show()



plot_scatter_shape(brcancer,cols)


def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
        plt.savefig('cond_hist.png')
    return grid_col


cond_hists(brcancer,cols, 'Diagnosis')

            
plot_scatter_shape(brcancer,cols)

#def cond_hists(df, plot_cols, grid_col):
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    for col in plot_cols:
#        grid1 = sns.FacetGrid(df, col=grid_col)
#        grid1.map(plt.hist, col, alpha=.7)
#        plt.savefig('cond_hist.png')
#    return grid_col
#
#cond_hists(brcancer,cols, 'Diagnosis')


cor_mat = brcancer.corr()
cor_mat['mean_radius'].sort_values(ascending=False)
brcancer_sc = pd.DataFrame()
for col in cols:
    brcancer_sc[col] = brcancer[col]
    if (col != 'mean_symmetry' and col != 'mean_smoothness' and col != 'worst_smoothness'):
        brcancer_sc[col] = np.log(brcancer[col])

brcancer_sc.replace([np.inf, -np.inf], 0, inplace=True)

brcancer_sc=brcancer
for col in cols:
    if (col != 'mean_symmetry' and col != 'mean_smoothness' and col!= 'worst_smoothness'):
        brcancer_sc[col] = np.cbrt(brcancer[col])

nr.seed(9988)
Features = np.array(brcancer_sc)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size=171)

x_train = Features[indx[0],2:]
x_test = Features[indx[1],2:]

# Rescale numeric features
scaler = preprocessing.StandardScaler().fit(x_train[:, 2:])
x_train[:, 2:] = scaler.transform(x_train[:, 2:])
x_test[:, 2:] = scaler.transform(x_test[:, 2:])

print(x_train.shape)
plot_density_hist(x_train,cols, bins=20, hist=True, name='scaled_dens_')

pca_mod = skde.PCA()
pca_comps = pca_mod.fit(x_train)
pca_comps
print(pca_comps.explained_variance_ratio_)
print(np.sum(pca_comps.explained_variance_ratio_))

def plot_explained(mod):
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    x = [y + 1 for y in x]          
    plt.plot(x,comps)

plot_explained(pca_comps)
pca_mod_5 = skde.PCA(n_components = 5)
pca_mod_5.fit(x_train)
Comps = pca_mod_5.transform(x_train)
Comps.shape
Comps_test=pca_mod_5.transform(x_test)

km_models = []
assignments_km = []
assignments_test_km = []
for i in range(2,5):
    
    kmeans_2 = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    assignments_km2 = kmeans_2.fit_predict(Comps)
    test_values = kmeans_2.predict(Comps_test)
    km_models.append(kmeans_2)
    assignments_km.append(assignments_km2)
    assignments_test_km.append(test_values)

 def create_labels(assignments_km, brcancer_indx):
     labels = np.zeros_like(assignments_km)
     for i in range(2):
         mask = (assignments_km == i)
         digits_mask = brcancer_indx[mask]
         labels[mask] = mode(digits_mask)[0]
     return labels
 
 #Training
 labels_training = create_labels(assignments_km[0], brcancer['Diagnosis'][indx[0]])
 labels_test = create_labels(assignments_test_km[0], brcancer['Diagnosis'][indx[1]])
 print(labels_training)
 print(labels_test)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Comps[:,0],Comps[:,1],c=labels_training , cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Comps[:,0],Comps[:,1],c = brcancer['Diagnosis'][indx[0]], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Comps_test[:,0],Comps_test[:,1],c=labels_test , cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Comps_test[:,0],Comps_test[:,1],c = brcancer['Diagnosis'][indx[1]], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')


def plot_clusters(sample, assignment):
    col_dic = {0:'blue',1:'red'}
    colors = [col_dic[x] for x in assignment]
    plt.scatter(sample[:,0],sample[:,1],color = colors)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Sample data')
    plt.show()

plot_clusters(Comps, labels_training)
plot_clusters(Comps_test,labels_test)

agglomerative_2 = AgglomerativeClustering(n_clusters=2)
assignments_ag2 = agglomerative_2.fit_predict(Comps)
labels_training_ag=create_labels(assignments_ag2,brcancer['Diagnosis'][indx[0]])
plot_clusters(Comps, assignments_ag2)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Comps[:,0],Comps[:,1],c=labels_training_ag , cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Comps[:,0],Comps[:,1],c = brcancer['Diagnosis'][indx[0]], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')

def plot_WCSS_km(km_models, sample):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    ## Plot WCSS
    wcss = [mod.inertia_ for mod in km_models]
    print(wcss)
    n_clusts = [x+1 for x in range(1,len(wcss) + 1)]
    ax[0].bar(n_clusts, wcss)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('WCSS')
    
    ## Plot BCSS
    tss = np.sum(sample[:,0:1]**2, axis = 0)
    print(tss)
    ## Compute BCSS as TSS - WCSS
    bcss = np.concatenate([tss - x for x in wcss]).ravel()
    ax[1].bar(n_clusts, bcss)
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('BCSS')
    plt.show()
    

plot_WCSS_km(km_models, Comps)


def plot_sillohette(samples, assignments, x_lab = 'Number of clusters', start =2):
    silhouette = [silhouette_score(samples[:,0:1], a) for a in assignments]
    n_clusts = [x + start for x in range(0, len(silhouette))]
    plt.bar(n_clusts, silhouette)
    plt.xlabel(x_lab)
    plt.ylabel('SC')
    plt.show()

plot_sillohette(Comps, assignments_km)
###########################################################
#ARI
from sklearn import metrics
labels_true = brcancer['Diagnosis'][indx[0]]
labels_pred = labels_training
metrics.adjusted_rand_score(labels_true, labels_pred)
metrics.adjusted_mutual_info_score(labels_true, labels_pred)  
metrics.normalized_mutual_info_score(labels_true, labels_pred)
metrics.homogeneity_score(labels_true, labels_pred)
metrics.completeness_score(labels_true, labels_pred)
metrics.v_measure_score(labels_true, labels_pred)
metrics.fowlkes_mallows_score(labels_true, labels_pred)
metrics.calinski_harabasz_score(Comps, labels_pred)
from sklearn.metrics.cluster import contingency_matrix
contingency_matrix(labels_true, labels_pred)
