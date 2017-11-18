# To run: just do 'python ensemble_submit.py'
# Note: paths variable must be specified.

import numpy as np
import pandas as pd

paths = ['submit4.txt']
# add paths here

default_path = './predictions/'

probs = np.loadtxt(default_path + paths[0])
for i in range(1, len(paths)):
    temp_prob = np.loadtxt(default_path + paths[i])
    probs += temp_prob

f = open('submit.txt', 'w+')
for i in range(10000):
    s = 'test/%08d.jpg' % (i + 1)
    tmp = probs[i,:]
    tmp2 = tmp.argsort()[-5:][::-1]
    for ans in tmp2:
        s += ' ' + str(ans)
    f.write(s + '\n')
    if i % 400 == 399:
        print(i + 1, " / 10000")
f.close()
