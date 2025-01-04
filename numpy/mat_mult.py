
import numpy as np

n = 20000 # 10000
arr1 = np.random.rand(n,n)
arr2 = np.random.rand(n,n)

#arr_result = np.multiply(arr1, arr2)  # serial
arr_result = np.matmul(arr1, arr2)     # multithreads
#print(arr_result)

