import numpy as np
n = [1,2,3,4,5,6,7,8,9,10]
print((n))
np.random.seed(0)
np.random.shuffle(np.array(n))
print((n))