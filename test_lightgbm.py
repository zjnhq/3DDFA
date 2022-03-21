from gradient_boosting import HistGradientBoostingRegressor
import numpy as np
n = 2000 
d= 10
x= np.random.randn(n,d)

y=np.random.randn(n)
print(y.shape)
print(x.shape)
model = HistGradientBoostingRegressor(max_iter=2, max_bins=255, max_leaf_nodes=31, warm_start = False)
model.fit(x,y)
n = 1000 
d= 10
x= np.random.randn(n,d)

y=np.random.randn(n)
param= model.get_params()
param['max_bins'] = 31
param['max_leaf_nodes'] = 63
param['max_iter']=20
param['warm_start'] = True
model.set_params(**param)
# model = HistGradientBoostingRegressor(max_iter=20, max_leaf_nodes=31, warm_start = True)
model.fit(x,y)
yb= model.predict(x)
