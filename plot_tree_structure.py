
"""
=======================================================================
Plot the decision surface of decision trees trained on the iris dataset
=======================================================================

Plot the decision surface of a decision tree trained on pairs
of features of the iris dataset.

See :ref:`decision tree <tree>` for more information on the estimator.

For each pair of iris features, the decision tree learns decision
boundaries made of combinations of simple thresholding rules inferred from
the training samples.

We also show the tree structure of a model built on all of the features.
"""
# %%
# First load the copy of the Iris dataset shipped with scikit-learn:
from sklearn.datasets import load_iris

iris = load_iris()

# %%
# Display the decision functions of trees trained on all pairs of features.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from gradient_boosting import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
np.random.seed(0)
plt.figure(figsize=(10, 10))
for fig_id in range(2):
	

	# for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
	for pairidx, pair in enumerate([[0, 1],  [0, 3], [2, 3]]):
		# We only take the two corresponding features
		X = iris.data[:, pair]
		y = iris.target

		# Train
		if fig_id==0:
			max_bins = 31
			max_leaf_nodes = 31
			max_iter =1
			clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
		if fig_id==1:
			max_bins = 3
			max_leaf_nodes = 3
			max_iter = 1
			clf = HistGradientBoostingClassifier(max_iter=max_iter, max_bins= max_bins, max_leaf_nodes = max_leaf_nodes)
			clf.fit(X,y)
			for stage in range(2):
				params= clf.get_params()
				params['max_iter'] = params['max_iter']+ 1
				params['warm_start'] = True
				params['max_bins'] = params['max_bins']+ max_bins +1
				params['max_leaf_nodes'] = params['max_leaf_nodes']+ max_leaf_nodes +1
				clf.set_params(**params)
				clf.fit(X,y)

		# Plot the decision boundary
		plt.subplot(2, 3, pairidx + 1 + int(fig_id*3))

		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(
			np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
		)
		plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

		plt.xlabel(iris.feature_names[pair[0]])
		plt.ylabel(iris.feature_names[pair[1]])

		# Plot the training points
		for i, color in zip(range(n_classes), plot_colors):
			idx = np.where(y == i)
			plt.scatter(
				X[idx, 0],
				X[idx, 1],
				c=color,
				label=iris.target_names[i],
				cmap=plt.cm.RdYlBu,
				edgecolor="black",
				s=15,
			)

	plt.suptitle("Comparison of decision boundary between standard decision tree (top) and CGBoost (bottom).")
	plt.legend(loc="lower right", borderpad=0, handletextpad=0)
	plt.tight_layout()
	_ = plt.axis("tight")
savefile = 'paper/iris_example'+str(fig_id)+'.pdf'
# plt.savefig(savefile, dpi=100)

# %%
# Display the structure of a single decision tree trained on all the features
# together.
from sklearn.tree import plot_tree
import pickle as pkl
plt.figure()
fileid= 3
filename = './gbdt_feature/gbdt_feature' +str(fileid)+'.pkl'
[X, y] = pkl.load(open(filename,'rb'))
N = 500
X=X[:N]
y=y[:N,-1]
clf = DecisionTreeRegressor(max_depth=3).fit(X, y)
plot_tree(clf, label= 'all', filled=True, fontsize=10)
plt.title("Decision tree for predicting coefficient of an expression component")
plt.show()
