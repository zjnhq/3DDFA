# from gradient_boosting import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pickle as pkl
# n = 2000 
# d= 10
# x= np.random.randn(n,d)

# y=np.random.randn(n)

filename = './gbdt_feature/gbdt_feature_layer6_12_file0.pkl'
[X, Y] = pkl.load(open(filename,'rb'))
N = X.shape[0]
d= 50
X = X[:N]
Y = Y[:N]
print(Y.shape)
print(X.shape)
max_leaf_nodes = 63
max_bins = 63
num_init_trees = 50
# end = time.time()
target_dim = Y.shape[1]
for i in range(56, target_dim):
	# print("training gbdt for target dim:"+str(i))
	# verbose = 1
	if i%15==0:
		verbose=1
	else:
		verbose =1
	verbose =1
	lr= 0.3
	# # if i>11:
	# 	lr= 0.2 
	# 	num_init_trees = 50
	model = HistGradientBoostingRegressor(max_iter=num_init_trees, learning_rate=lr, max_leaf_nodes=max_leaf_nodes, max_bins=max_bins, warm_start = False, verbose=verbose, validation_fraction=0.3)
	# lightgbm = GradientBoostingRegressor(n_estimators=20, max_depth=6)
	# model = HistGradientBoostingRegressor(max_iter=10, max_bins=31, max_leaf_nodes=31, verbose = 2, warm_start = False, validation_fraction=0.2)
	x = X
	y= Y[:,i]
	model.fit(x,y)
	print(model.score(x, y))
	# filename = './gbdt_feature/gbdt_feature1.pkl'
	# [x, y] = pkl.load(open(filename,'rb'))
	# x = x[:N]
	# y = y[:N,d]
	# param= model.get_params()
	# param['max_bins'] = 63
	# param['max_leaf_nodes'] = 63
	# param['max_iter']=20
	# param['warm_start'] = True
	# model.set_params(**param)
	# # model = HistGradientBoostingRegressor(max_iter=20, max_leaf_nodes=31, warm_start = True)
	# model.fit(x,y)
	# # yb= model.predict(x)
	# print(model.score(x, y))
