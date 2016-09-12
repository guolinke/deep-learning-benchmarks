import numpy as np
dim = 26752
a = range(0, dim)
np.savetxt('./labelmap26752.txt', np.reshape(a, (dim, 1)), fmt='%d')
