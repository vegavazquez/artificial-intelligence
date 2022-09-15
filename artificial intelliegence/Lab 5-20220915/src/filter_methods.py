#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

from sklearn.feature_selection import f_regression, mutual_info_regression, f_classif, mutual_info_classif

def example_one():

    np.random.seed(0)
    X = np.random.rand(1000, 3)
    y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

    f_test, _ = f_regression(X, y)
    f_test /= np.max(f_test)

    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(X[:, i], y, edgecolor='black', s=20)
        plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
        if i == 0:
            plt.ylabel("$y$", fontsize=14)
        plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),fontsize=16)

    plt.show()

    return X[:,0], X[:,1], X[:,2], y


def example_two(mu = 2.5, sigma_1 = 2, sigma_2 = 5):
    
    # toy example: linearly separable problem
    np.random.seed(0)

    # -- parameters
    N      = 1000

    # -- variables auxiliares
    unos  = np.ones(int(N/2))
    rand2 = np.random.randn(int(N/2),1)

    # -- features
    y  = np.concatenate([-1*unos,unos]) 

    X1 = np.concatenate([-mu + sigma_1*rand2,mu + sigma_1*rand2])
    X2 = sigma_2*np.random.randn(N,1) - 3*X1
    X3 = 2*np.random.randn(N,1)
    X4 = np.random.rand(N,1)

    X  = np.hstack((X1,X2,X3,X4))

    plt.figure(figsize=(20, 4))

    for i in range(4):
        plt.subplot(1, 5, i + 1)
        plt.hist(X[y<0,i],bins=20, density=True, alpha=0.5, label='-1',color='b')
        plt.hist(X[y>0,i],bins=20, density=True, alpha=0.5, label='+1',color='r')
        plt.legend(loc='upper right')
        plt.xlabel("$x_{}$".format(i + 1), fontsize=18)

    plt.subplot(1, 5, 5)
    plt.scatter(X[:,0],X[:,1], c=y, cmap = cm_bright, alpha=0.5)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18)

    plt.show()

    print('CORRELATION MATRIX:')
    display(pd.DataFrame(data=X, columns = ['x1', 'x2', 'x3', 'x4']).corr())

    return X, y

def example_three(mu = 1.5, sigma=1.5):
    np.random.seed(0)

    # -- parameters
    N     = 1000
    #mu    = 1.5      # Cambia este valor
    #sigma = 1.5      # Cambia este valor

    # variables auxiliares
    unos = np.ones(int(N/4))
    random4 = sigma*np.random.randn(int(N/4),1)
    random2 = sigma*np.random.randn(int(N/2),1)

    # -- features
    y = np.concatenate([-1*unos,       unos,          unos,         -1*unos]) 
    X1 = np.concatenate([-mu + random4, mu + random4, -mu + random4, mu + random4])
    X2 = np.concatenate([+mu + random2,               -mu + random2])

    X3 = 3*(X1+X2) + np.sqrt(2)*np.random.randn(N,1)
    X4 = 2*np.square((X1+X2)) + np.sqrt(2)*np.random.randn(N,1)

    E  = 2*np.random.randn(N, 20) # noisy variables

    X  = np.hstack((X1,X2,X3,X4,E))

    # do the plotting
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 4, 1)

    plt.scatter(X1,X2,c=y, cmap=cm_bright)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)

    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.hist(X[y<0,i+1],bins=20, density=True, alpha=0.5, label='-1',color='b')
        plt.hist(X[y>0,i+1], bins=20, density=True, alpha=0.5, label='+1',color='r')
        plt.legend(loc='upper right')
        plt.xlabel("$x_{}$".format(i + 2), fontsize=18)

    plt.show()
    
    return X, y

def filter_methods_classification(X, y, feat_names, rotation=False):
    
    angle = 0
    if rotation:
        angle = 90
        
    # do calculations
    f_test, _ = f_classif(X, y)
    f_test /= np.max(f_test)

    mi = mutual_info_classif(X, y)
    mi /= np.max(mi)

    # do some plotting
    plt.figure(figsize=(20, 4))

    plt.subplot(1,2,1)
    plt.bar(range(X.shape[1]),f_test,  align="center")
    plt.xticks(range(X.shape[1]),feat_names, rotation = angle)
    plt.xlabel('features')
    plt.ylabel('Ranking')
    plt.title('$F-test$ (ANOVA) score')

    plt.subplot(1,2,2)
    plt.bar(range(X.shape[1]),mi,  align="center")
    plt.xticks(range(X.shape[1]),feat_names, rotation = angle)
    plt.xlabel('features')
    plt.ylabel('Ranking')
    plt.title('Mutual information score')

    plt.show()