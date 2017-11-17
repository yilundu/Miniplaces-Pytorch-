# To run: just do 'python ensemble_submit.py'
# Note: paths variable must be specified.

import numpy as np
import pandas as pd

paths = ['submit-1.csv', 'submit-1.csv']
# add paths here

default_path = './predictions/'

probs = pd.read_csv(default_path + paths[0])
for i in range(1, len(paths)):
	probs += pd.read_csv(default_path + paths[i])

f = open('submit.txt', 'w+')
for i in range(10000):
    s = 'test/%08d.jpg' % (i + 1)
    tmp = probs.iloc[i,1:].as_matrix()
    tmp2 = tmp.argsort()[-5:][::-1]
    for ans in tmp2:
        s += ' ' + str(ans)
    f.write(s + '\n')
    if i % 400 == 399:
        print(i + 1, " / 10000")
f.close()