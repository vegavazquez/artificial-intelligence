#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def toy_example_pca():
	
	rng = np.random.RandomState(1)
	X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

	# do the plotting
	plt.scatter(X[:, 0], X[:, 1])
	plt.xlabel('$x_1$', fontsize=16)
	plt.ylabel('$x_2$', fontsize=16)
	plt.axis('equal');

	return X


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def plot_pca_toy_example(X):

	#pca = PCA(n_components=2, whiten=True).fit(X)
	pca = PCA(n_components=2).fit(X)
	X_pca = pca.transform(X)

	fig, ax = plt.subplots(1, 3, figsize=(20, 5))
	fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

	# plot data
	ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
	for length, vector in zip(pca.explained_variance_, pca.components_):
	    v = vector * 3 * np.sqrt(length)
	    draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
	ax[0].axis('equal');
	ax[0].set(xlabel='x', ylabel='y', title='input')

	# plot principal components
	ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
	draw_vector([0, 0], [0, 1], ax=ax[1])
	draw_vector([0, 0], [3, 0], ax=ax[1])
	ax[1].axis('equal')
	ax[1].set(xlabel='component 1', ylabel='component 2',
	          title='principal components',
	          xlim=(-5, 5), ylim=(-3, 3.1))

	# plot principal components
	pca = PCA(n_components=2, whiten=True).fit(X)
	X_pca = pca.transform(X)

	ax[2].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
	draw_vector([0, 0], [0, 3], ax=ax[2])
	draw_vector([0, 0], [3, 0], ax=ax[2])
	ax[2].axis('equal')
	ax[2].set(xlabel='component 1', ylabel='component 2',
	          title='principal components (whiten: unit variance)',
	          xlim=(-5, 5), ylim=(-3, 3.1))


	return pca
