#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc

def get_metrics(confmat):
    '''Unravel confusion matrix and calculate performance metrics'''
    tn, fp, fn, tp = confmat.ravel()
    
    acc = (tp+tn)/(tn + fp + fn + tp)
    sen = tp/(tp+fn)
    esp = tn/(tn+fp)
    ppv = tp/(tp+fp)
    fsc = 2*(sen*ppv)/(sen+ppv)
    
    return acc, sen, esp, ppv, fsc
    
def print_performance_metrics(confmat_train, *confmat_test):
    '''Print performance metrics'''
    
    if not confmat_test:
        acc, sen, esp, ppv, fsc = get_metrics(confmat_train)
        print('ACC: %2.2f' %(100*acc))
        print('SEN: %2.2f' %(100*sen))
        print('ESP: %2.2f' %(100*esp))
        print('PPV: %2.2f' %(100*ppv))
        print('F1: %2.2f' %(100*fsc))
    else:
        acc_train, sen_train, esp_train, ppv_train, fsc_train = get_metrics(confmat_train)
        acc_test, sen_test, esp_test, ppv_test, fsc_test = get_metrics(confmat_test[0])
        
        print('PERFORMANCE METRICS')
        print('\tTRAIN\tTEST')
        print('ACC:\t%2.2f\t%2.2f' %(100*acc_train, 100*acc_test))
        print('SEN:\t%2.2f\t%2.2f' %(100*sen_train, 100*sen_test))
        print('ESP:\t%2.2f\t%2.2f' %(100*esp_train, 100*esp_test))
        print('PPV:\t%2.2f\t%2.2f' %(100*ppv_train, 100*ppv_test))
        print('F1:\t%2.2f\t%2.2f'  %(100*fsc_train, 100*fsc_test))
        
def plot_roc_curve(y,y_prob):
    
    '''Plot ROC-AUC Curve and target probability'''
    
    ejex, ejey, _ = roc_curve(y, y_prob)
    roc_auc = auc(ejex, ejey)

    plt.figure(figsize = (12,4))
    
    # ROC-AUC CURVE
    plt.subplot(1,2,1)
    plt.plot(ejex, ejey, color='darkorange',lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color=(0.6, 0.6, 0.6), linestyle='--')
    plt.plot([0, 0, 1],[0, 1, 1],lw=2, linestyle=':',color='black',label='Perfect classifier')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR (1-ESP)')
    plt.ylabel('SEN')
    plt.legend(loc="lower right")
    
    # PROB DENSITY 
    idx_0 = (y==0)
    idx_1 = (y==1)
    
    plt.subplot(1,2,2)
    plt.hist(y_prob[idx_0],density=1,bins = 20, label='y=0',alpha=0.5)
    plt.hist(y_prob[idx_1],density=1,bins = 20, facecolor='red',label='y=1',alpha=0.5)
    plt.legend()
    plt.xlabel('target probability')
    
    plt.show()

def plot_confusion_matrix(confmat_train, *confmat_test):
    ''' Plot confusion matrix
        - A single confusion matrix
        - Comparing two confusion matrices, if provided
    '''
    
    if not confmat_test:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.matshow(confmat_train, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_train.shape[0]):
            for j in range(confmat_train.shape[1]):
                ax.text(x=j, y=i, s=confmat_train[i, j], va='center', ha='center')

        plt.xlabel('predicted label')
        plt.ylabel('true label')

        plt.tight_layout()
        plt.show()
        
    else:
        fig, ax = plt.subplots(1,2,figsize=(6, 6))
        ax[0].matshow(confmat_train, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_train.shape[0]):
            for j in range(confmat_train.shape[1]):
                ax[0].text(x=j, y=i, s=confmat_train[i, j], va='center', ha='center')

        ax[1].matshow(confmat_test[0], cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_test[0].shape[0]):
            for j in range(confmat_test[0].shape[1]):
                ax[1].text(x=j, y=i, s=confmat_test[0][i, j], va='center', ha='center')
    
        ax[0].set_xlabel('predicted label')
        ax[0].set_ylabel('true label')
        ax[0].set_title('TRAIN')
        
        ax[1].set_xlabel('predicted label')
        ax[1].set_ylabel('true label')
        ax[1].set_title('TEST')

        plt.tight_layout()
        plt.show()

def analyze_train_test_performance(clf, X_train, X_test, y_train, y_test):
    
    '''Analyze Train and Test Performance'''
    
    # get predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)
    
    # get confusion matrices
    confmat_train = confusion_matrix(y_train, y_pred_train)
    confmat_test  = confusion_matrix(y_test, y_pred_test)
    
    # Plot confusion matrices and provide metrics
    print_performance_metrics(confmat_train, confmat_test)
    plot_confusion_matrix(confmat_train, confmat_test)

    # Plot ROC curve
    y_prob = clf.predict_proba(X_test)[:,1]
    plot_roc_curve(y_test,y_prob)

def hyper_parameters_search_classification(clf, X, y, param_grid, scorer = 'accuracy', cv=3):
    
    grid = GridSearchCV(clf, param_grid = param_grid, scoring = scorer, cv = cv)
    grid.fit(X, y)

    print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("best parameters: {}".format(grid.best_params_))
    
    return grid


def hyper_parameters_search_regression(clf, X, y, param_grid, scorer = 'neg_mean_squared_error', cv=3):
    
    grid = GridSearchCV(clf, param_grid = param_grid, scoring = scorer, cv = cv)
    grid.fit(X, y)

    print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("best parameters: {}".format(grid.best_params_))
    
    return grid

def plot_cv_scoring_classification(grid, hyper_parameter, log=False):
    
    scores = np.array(grid.cv_results_['mean_test_score'])
    params = grid.param_grid[hyper_parameter]
    
    if log:
        params = np.log10(params)
        
    plt.plot(params,scores,'-o')
    plt.xlabel(hyper_parameter,fontsize=14)
    plt.ylabel(grid.get_params()['scoring'])
    plt.show()

def plot_cv_scoring_regression(grid, hyper_parameter, scorer = 'MSE', log=False):
    
    scores = -1*np.array(grid.cv_results_['mean_test_score'])
    params = grid.param_grid[hyper_parameter]
    
    if scorer == 'r2':
        scores = -1*scores
    if log:
        params = np.log10(params)
        
    plt.plot(params,scores,'-o')
    plt.xlabel(hyper_parameter,fontsize=14)
    plt.ylabel(grid.get_params()['scoring'])
    plt.show()


def plot_importances(importances, feat_names):
    
    df_importances = pd.Series(importances, index=feat_names)
    
    plt.figure(figsize=(9,4))
    df_importances.plot.bar()
    plt.ylabel("Feature Importance")
    plt.show()