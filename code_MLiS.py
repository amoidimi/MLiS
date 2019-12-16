# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 01:17:50 2019

@author: dimit
"""

#Visualising distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
brcancer = pd.read_csv('data.csv')


def clean_auto_data(brcancer):
    import pandas as pd
    import numpy as np
    cols = brcancer.columns
    brcancer.columns = [str.replace(' ', '_') for str in cols]
    ## Transform column data type
    ## Convert some columns to numeric values
    cols = ['mean_radius', 'mean_texture', 'mean_perimeter','mean_area','mean_smoothness',
            'mean_compactness','mean_concavity','mean_concave_points','mean_symmetry','mean_fractal_dimension',
            'std_radius', 'std_texture', 'std_perimeter','std_area','std_smoothness',
            'std_compactness','std_concavity','std_concave_points','std_symmetry','std_fractal_dimension',
            'worst_radius', 'worst_texture', 'worst_perimeter','worst_area','worst_smoothness',
            'worst_compactness','worst_concavity','worst_concave_points','worst_symmetry','worst_fractal_dimension']
    for column in cols:
        brcancer.loc[brcancer[column] == '?', column] = np.nan
    brcancer.dropna(axis = 0, inplace = True)
    for column in cols:
        brcancer[column] = pd.to_numeric(brcancer[column])

    return brcancer

brcancer = clean_auto_data(brcancer)
print(brcancer.columns)
brcancer.head()
brcancer.describe()
brcancer.info()

def plot_histogram(brcancer, cols, bins):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) 
        ax = fig.gca()   
        brcancer[col].plot.hist(ax = ax, bins = bins)
        ax.set_title('Histogram of ' + col)
        ax.set_xlabel(col) 
        ax.set_ylabel('Number of patients')
        plt.show()
        
num_cols =['mean_radius', 'mean_texture', 'mean_perimeter','mean_area','mean_smoothness',
           'mean_compactness','mean_concavity','mean_concave_points','mean_symmetry','mean_fractal_dimension']    
plot_histogram(brcancer, num_cols,10)

def plot_density_hist(brcancer, cols, bins, hist):
   for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(brcancer[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col)
        plt.xlabel(col) 
        plt.ylabel('Number of patients')
        plt.show()
        
plot_density_hist(brcancer, num_cols,bins=20,hist = False)   
plot_density_hist(brcancer, num_cols, bins = 20, hist = True)  

def plot_scatter(brcancer, cols, col_y = 'mean_radius'):
    for col in cols:
        fig = plt.figure(figsize=(7,6))
        ax = fig.gca()   
        brcancer.plot.scatter(x = col, y = col_y, ax = ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) 
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()

plot_scatter(brcancer, num_cols) 

plot_scatter(brcancer, ['mean_texture'], 'mean_perimeter') 

def plot_scatter_t(brcancer, cols, col_y = 'mean_radius', alpha = 1.0):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        brcancer.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) 
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()
       # fig.savefig('scatterplot.png')

plot_scatter_t(brcancer, num_cols, alpha = 0.2)   

def plot_desity_2d(brcancer, cols, col_y = 'mean_radius', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=brcancer, kind=kind)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

plot_desity_2d(brcancer, num_cols) 
plot_desity_2d(brcancer, num_cols, kind = 'hex')

def plot_scatter_shape(brcancer, cols, shape_col = 'Diagnosis', col_y = 'mean_radius', alpha = 0.2):
    shapes = ['o','x'] 
    unique_cats = brcancer[shape_col].unique()
    for col in cols: 
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): 
            temp = brcancer[brcancer[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.legend()
        plt.show()
            
plot_scatter_shape(brcancer, num_cols)

def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
        plt.savefig('cond_hist.png')
    return grid_col

cond_hists(brcancer, num_cols, 'Diagnosis')

cor_mat=brcancer.corr()
cor_mat['mean_radius'].sort_values(ascending=False)

