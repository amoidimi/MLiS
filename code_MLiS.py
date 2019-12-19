# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 01:17:50 2019

@author: dimit
"""

# Visualising distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nr
import math
from sklearn import metrics
from scipy.stats import mode
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.decomposition as skde
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix

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

#plot_histogram(brcancer, cols, 10)

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
        
#plot_density_hist(brcancer,cols, bins = 20, hist = True,name='density_')


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

#plot_scatter(brcancer,cols)

#plot_scatter(brcancer, ['mean_texture'], 'mean_perimeter')


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


#plot_scatter_t(brcancer,cols, alpha=0.2)


def plot_desity_2d(brcancer, cols, col_y='mean_radius', kind='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=brcancer, kind=kind)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

#plot_desity_2d(brcancer,cols)
#plot_desity_2d(brcancer,cols, kind='hex')


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



#plot_scatter_shape(brcancer,cols)


def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
        plt.savefig('cond_hist.png')
    return grid_col


#cond_hists(brcancer,cols, 'Diagnosis')

            
#plot_scatter_shape(brcancer,cols)

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

brcancer_sc = pd.DataFrame()
for col in cols:
    brcancer_sc[col] = brcancer[col]
    if (col != 'mean_symmetry' and col != 'mean_smoothness' and col != 'worst_smoothness'):
        brcancer_sc[col] = np.log(brcancer[col])

brcancer_sc.replace([np.inf, -np.inf], 0, inplace=True)

nr.seed(9988)
Features = np.array(brcancer_sc)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size=114)

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

# number of components
n_pcs = pca_mod_5.components_.shape[0]

most_important = [5, 13, 6, 9, 19, 14]
def most_important_features(comps):
    features = []
    for j in most_important:
        features.append(abs(comps[j]))

    return features

pca_comp_values = {}
initial_feature_names = cols
for i in range(n_pcs):
    comps = pca_mod_5.components_[i]
    idx = 'PCA'+ str(i);
    pca_comp_values[idx] = most_important_features(comps)

# PCA heat plot
plt.matshow(pca_mod_5.components_,cmap='coolwarm')
plt.yticks([0,1,2,3,4],['1st Comp','2nd Comp','3rd Comp', '4th comp', '5th comp'],fontsize=9)
plt.colorbar()
plt.xticks(range(len(cols)),cols,rotation=65,ha='left',fontsize=9)
plt.tight_layout()
plt.show()

# PCA plot
# Setting the positions and width for the bars
pos = list(range(len(pca_comp_values['PCA0'])))
width = 0.1
palette = sns.diverging_palette(220, 20, n = 8)

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        pca_comp_values['PCA0'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=palette[0],
        # with label the first value in first_name
        label=cols[most_important[0]])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        pca_comp_values['PCA1'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=palette[2],
        # with label the second value in first_name
        label=cols[most_important[1]])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width * 2 for p in pos],
        # using df['post_score'] data,
        pca_comp_values['PCA2'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=palette[5],
        # with label the third value in first_name
        label=cols[most_important[2]])

plt.bar([p + width * 3 for p in pos],
        # using df['post_score'] data,
        pca_comp_values['PCA3'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=palette[6],
        # with label the third value in first_name
        label=cols[most_important[3]])

plt.bar([p + width * 4 for p in pos],
        # using df['post_score'] data,
        pca_comp_values['PCA4'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=palette[7],
        # with label the third value in first_name
        label=cols[most_important[4]])

# Set the y axis label
ax.set_ylabel('Importance Score')

# Set the chart's title
ax.set_title('Features')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(cols[most_important[i]] for i in range(0, len(most_important)))

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 5)
plt.ylim([0, max(pca_comp_values['PCA0'] + pca_comp_values['PCA1'] + pca_comp_values['PCA2'] + pca_comp_values['PCA3'] + pca_comp_values['PCA4'])])

# Adding the legend and showing the plot
plt.legend(['PCA0', 'PCA1', 'PCA2', 'PCA3', 'PCA4'], loc='upper left')
plt.grid()
plt.show()

km_models = []
assignments_km = []
assignments_test_km = []
for i in range(2,5):
    
    kmeans= KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    assignments= kmeans.fit_predict(Comps)
    test_values = kmeans.predict(Comps_test)
    km_models.append(kmeans)
    assignments_km.append(assignments)
    assignments_test_km.append(test_values)

def create_labels(assign,brcancer_indx):
     labels = np.zeros_like(assign)
     for i in range(2):
         mask = (assign == i)
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
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Comps_test[:,0],Comps_test[:,1],c=labels_test , cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Comps_test[:,0],Comps_test[:,1],c = brcancer['Diagnosis'][indx[1]], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')

plt.show()
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

def plot_sillohette(samples, assignments, x_lab = 'Number of clusters', start =2):
    silhouette = [silhouette_score(samples[:,0:1], a) for a in assignments]
    n_clusts = [x + start for x in range(0, len(silhouette))]
    plt.bar(n_clusts, silhouette)
    plt.xlabel(x_lab)
    plt.ylabel('SC')
    plt.show()

plot_sillohette(Comps, assignments_km)
###########################################################

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

contingency_matrix(labels_true, labels_pred)

#Cross validation

k_values = [2, 3, 4, 5, 6]
nk = len(k_values)
lims=np.arange(0,568,57)
lims=np.append(lims,569)
print(lims)
u=1
ARI=np.zeros(shape=(10, nk))
AMI=np.zeros(shape=(10,nk))
NMI=np.zeros(shape=(10,nk))
H=np.zeros(shape=(10,nk))
C=np.zeros(shape=(10,nk))
VHC=np.zeros(shape=(10,nk))
FMS=np.zeros(shape=(10,nk))
CHS=np.zeros(shape=(10,nk))
SIL=np.zeros(shape=(10,nk))

pca_features = []
for k in range(len(lims)-1):
    n = len(lims) - 1
    Features = np.array(brcancer_sc)
    x_train = Features[lims[0]:lims[n],:]
    x_test = Features[lims[n-u]:lims[n-u+1],:]
    train_set = np.arange(lims[0], lims[n], 1)
    test_set=np.arange(lims[n-u],lims[n-u+1],1)
    train_set = np.delete(train_set, test_set, 0)
    x_train=np.delete(x_train,test_set,0)
    u=u+1
    print(k)
    
    # Rescale numeric features,
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train= scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    pca_mod_5 = skde.PCA(n_components = 5)
    pca_mod_5.fit(x_train)
    Comps = pca_mod_5.transform(x_train)
    Comps_test=pca_mod_5.transform(x_test)

    for i in k_values:
        print('For k: {}'.format(i))
        kmeans= KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
        assignments_km = kmeans.fit_predict(Comps)
        assignments_test_km = kmeans.predict(Comps_test)

        labels_true = brcancer['Diagnosis'][train_set]
        labels_pred = assignments_km
        ARI[k,i-2]=metrics.adjusted_rand_score(labels_true, labels_pred)
        AMI[k,i-2]=metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        NMI[k,i-2]=metrics.normalized_mutual_info_score(labels_true, labels_pred)
        H[k,i-2]=metrics.homogeneity_score(labels_true, labels_pred)
        C[k,i-2]=metrics.completeness_score(labels_true, labels_pred)
        VHC[k,i-2]=metrics.v_measure_score(labels_true, labels_pred)
        FMS[k,i-2]=metrics.fowlkes_mallows_score(labels_true, labels_pred)
        CHS[k,i-2]=metrics.calinski_harabasz_score(Comps, labels_pred)
        SIL[k,i-2]=metrics.silhouette_score(Comps[:,0:1],labels_pred)

performance_scores = {}

def calculate_mean(scores):
    means = []
    for i in range(0,5):
        mean = np.mean(scores[:, i])
        means.append(mean)
    return means


performance_scores['ARI'] = calculate_mean(ARI)
performance_scores['AMI'] = calculate_mean(AMI)
performance_scores['NMI'] = calculate_mean(NMI)
performance_scores['H'] = calculate_mean(H)
performance_scores['C'] = calculate_mean(C)
performance_scores['VHC'] = calculate_mean(VHC)
performance_scores['FMS'] = calculate_mean(FMS)
performance_scores['CHS'] = calculate_mean(CHS)
performance_scores['SIL'] = calculate_mean(SIL)

df=pd.DataFrame({'x': k_values, 'SIL': performance_scores['SIL'],
                 'FMS': performance_scores['FMS'],
                 'ARI': performance_scores['ARI'],
                 'NMI': performance_scores['NMI']})
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

num=0
for column in df.drop('x', axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
plt.legend(loc = 1)
plt.title("Average Performance scores after cross validation", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Number of clusters")
plt.ylabel("Performance Score")
plt.show()

# plt.plot(k_values, performance_scores['SIL'], 'r', k_values, performance_scores['FMS'], 'b', k_values, performance_scores['ARI'], 'g')
# plt.show()

############################################################
#hierarchical

agg_models = []
assignments_agg = []
for i in range(2,5):
    
    agg=AgglomerativeClustering(n_clusters=i)
    assignments= agg.fit_predict(Comps)
    agg_models.append(agg)
    assignments_agg.append(assignments)
    
plot_clusters(Comps, assignments_agg[0])
labels_training_agg = create_labels(assignments_agg[0], brcancer['Diagnosis'][indx[0]])

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Comps[:,0],Comps[:,1],c=labels_training_agg , cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('Hierarchical clustering plot')

ax2.scatter(Comps[:,0],Comps[:,1],c = brcancer['Diagnosis'][indx[0]], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')
plt.show()

plot_sillohette(Comps, assignments_agg)
###########################################################

labels_true = brcancer['Diagnosis'][indx[0]]
labels_pred = labels_training_agg
metrics.adjusted_rand_score(labels_true, labels_pred)
metrics.adjusted_mutual_info_score(labels_true, labels_pred)  
metrics.normalized_mutual_info_score(labels_true, labels_pred)
metrics.homogeneity_score(labels_true, labels_pred)
metrics.completeness_score(labels_true, labels_pred)
metrics.v_measure_score(labels_true, labels_pred)
metrics.fowlkes_mallows_score(labels_true, labels_pred)
metrics.calinski_harabasz_score(Comps, labels_pred)

contingency_matrix(labels_true, labels_pred)

#Cross validation

k_values = [2, 3, 4, 5, 6]
nk = len(k_values)
lims=np.arange(0,568,57)
lims=np.append(lims,569)
print(lims)
u=1
ARI=np.zeros(shape=(10, nk))
AMI=np.zeros(shape=(10,nk))
NMI=np.zeros(shape=(10,nk))
H=np.zeros(shape=(10,nk))
C=np.zeros(shape=(10,nk))
VHC=np.zeros(shape=(10,nk))
FMS=np.zeros(shape=(10,nk))
CHS=np.zeros(shape=(10,nk))
SIL=np.zeros(shape=(10,nk))

pca_features = []
for k in range(len(lims)-1):
    n = len(lims) - 1
    Features = np.array(brcancer_sc)
    x_train = Features[lims[0]:lims[n],:]
    x_test = Features[lims[n-u]:lims[n-u+1],:]
    train_set = np.arange(lims[0], lims[n], 1)
    test_set=np.arange(lims[n-u],lims[n-u+1],1)
    train_set = np.delete(train_set, test_set, 0)
    x_train=np.delete(x_train,test_set,0)
    u=u+1
    print(k)
    
    # Rescale numeric features,
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train= scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    pca_mod_5 = skde.PCA(n_components = 5)
    pca_mod_5.fit(x_train)
    Comps = pca_mod_5.transform(x_train)
    Comps_test=pca_mod_5.transform(x_test)

    for i in k_values:
        print('For k: {}'.format(i))
        agg= AgglomerativeClustering(n_clusters=i)
        assignments_agg = agg.fit_predict(Comps)

        labels_true = brcancer['Diagnosis'][train_set]
        labels_pred = assignments_agg
        ARI[k,i-2]=metrics.adjusted_rand_score(labels_true, labels_pred)
        AMI[k,i-2]=metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        NMI[k,i-2]=metrics.normalized_mutual_info_score(labels_true, labels_pred)
        H[k,i-2]=metrics.homogeneity_score(labels_true, labels_pred)
        C[k,i-2]=metrics.completeness_score(labels_true, labels_pred)
        VHC[k,i-2]=metrics.v_measure_score(labels_true, labels_pred)
        FMS[k,i-2]=metrics.fowlkes_mallows_score(labels_true, labels_pred)
        CHS[k,i-2]=metrics.calinski_harabasz_score(Comps, labels_pred)
        SIL[k,i-2]=metrics.silhouette_score(Comps[:,0:1],labels_pred)

performance_scores = {}

def calculate_mean(scores):
    means = []
    for i in range(0,5):
        mean = np.mean(scores[:, i])
        means.append(mean)
    return means


performance_scores['ARI'] = calculate_mean(ARI)
performance_scores['AMI'] = calculate_mean(AMI)
performance_scores['NMI'] = calculate_mean(NMI)
performance_scores['H'] = calculate_mean(H)
performance_scores['C'] = calculate_mean(C)
performance_scores['VHC'] = calculate_mean(VHC)
performance_scores['FMS'] = calculate_mean(FMS)
performance_scores['CHS'] = calculate_mean(CHS)
performance_scores['SIL'] = calculate_mean(SIL)

df=pd.DataFrame({'x': k_values, 'SIL': performance_scores['SIL'],
                 'FMS': performance_scores['FMS'],
                 'ARI': performance_scores['ARI'],
                 'NMI': performance_scores['NMI']})
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

num=0
for column in df.drop('x', axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
plt.legend(loc = 1)
plt.title("Average Performance scores after cross validation", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Number of clusters")
plt.ylabel("Performance Score")
plt.show()
#############################################################
#from sklearn.cluster import AffinityPropagation
#import matplotlib.pyplot as plt
#
## Setup Affinity Propagation
#
#aff = AffinityPropagation(damping=0.7, max_iter=10000, convergence_iter=100, 
#                          copy=True, preference=None, affinity='euclidean', verbose=False)
#assignments_aff = aff.fit_predict(Comps)
#test_values_aff = aff.predict(Comps_test)
#centers=aff.cluster_centers_indices_
#n_clusters_aff=len(centers)
#labels=aff.labels_
#
#print('Estimated number of clusters: %d' % n_clusters_aff)
## Plot exemplars
#
#plt.close('all')
#plt.figure(1)
#plt.clf()
#
#for k, col in zip(range(n_clusters_aff), palette):
#    class_members = labels == k
#    cluster_center = Comps[centers[k]]
#    plt.plot(Comps[class_members, 0], Comps[class_members, 1], palette + '.')
#    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=palette, markeredgecolor='k', markersize=14)
#    for x in Comps[class_members]:
#        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
#plt.show()