import numpy as np

val_split = 0.2
with open('all.txt') as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
for img in lines[num_train:]: # lines[:num_train]
    print(img, end='', flush=True)