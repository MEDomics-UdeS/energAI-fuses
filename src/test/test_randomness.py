import sys
import random
import datetime as dt

import numpy as np
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

features = torch.randn(2, 5)

# Print stuff.
fnp = features.view(-1).numpy()

print("Time: {}".format(dt.datetime.now()))
for el in fnp:
    print("{:.20f}".format(el))

print("Python: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Pytorch: {}".format(torch.__version__))