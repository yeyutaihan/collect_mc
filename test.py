import numpy as np

path = "/home/tc0786/Project/collect_mc/eval_data/additional_mem/000000.npz"

data = np.load(path)["actions"]
print(data.shape)