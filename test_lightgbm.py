from gradient_boosting import HistGradientBoostingRegressor
# from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pickle as pkl
# n = 2000 
# d= 10
# x= np.random.randn(n,d)

# y=np.random.randn(n)

filename = './gbdt_feature/gbdt_feature0.pkl'
[x, y] = pkl.load(open(filename,'rb'))
N = 2000
d= 20
x = x[:N]
y = y[:N,d]
print(y.shape)
print(x.shape)
model = HistGradientBoostingRegressor(max_iter=10, max_bins=31, max_leaf_nodes=31, verbose = 2, warm_start = False, validation_fraction=0.2)
model.fit(x,y)
print(model.score(x, y))
filename = './gbdt_feature/gbdt_feature1.pkl'
[x, y] = pkl.load(open(filename,'rb'))
x = x[:N]
y = y[:N,d]
param= model.get_params()
param['max_bins'] = 63
param['max_leaf_nodes'] = 63
param['max_iter']=20
param['warm_start'] = True
model.set_params(**param)
# model = HistGradientBoostingRegressor(max_iter=20, max_leaf_nodes=31, warm_start = True)
model.fit(x,y)
# yb= model.predict(x)
print(model.score(x, y))
