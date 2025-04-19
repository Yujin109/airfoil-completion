import numpy as np

a = [(2, 249), (1,), (1,)]

#Count number of elements within a
count = sum([np.prod(x) for x in a])

print(count)