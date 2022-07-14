import torch
import time
import sys

sys.path.append("..")
import LoadFashionMNIST as lf

train_iter, test_iter = lf.get_iter()
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


