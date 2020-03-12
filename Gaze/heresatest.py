import numpy as np


A=np.ones(shape=(1,64,64,3))

print A.shape[1]
print np.shape(A)

print A


B=np.pad(A,((2,2), (2,2),(0,0)), 'constant', constant_values = (0,0))

print np.shape(B)
