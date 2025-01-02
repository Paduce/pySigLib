import pysiglib as sig
import numpy as np
import iisignature
import time
import matplotlib.pyplot as plt
import signatory
import esig
from tqdm import tqdm
import torch

if __name__ == '__main__':
    sig.cpsig_hello_world(5)
    sig.cusig_hello_world(5)
    print("\n")
    X = np.random.randint(low = 0, high = 10, size = (10000, 5)).astype("double")
    X_ = np.random.randint(low=0, high=10, size=(1, 100000, 5)).astype("double")
    torchX = torch.tensor(X_)
    #X = np.array([[0,0],[0.5,0.5],[1,1]]).astype("double")
    #X = np.array([[0, 0], [0.25, 0.25], [0.75, 0.75], [1,1]]).astype("double")
    #X = np.array([[0, 0], [1, 0.5], [4, 0], [0, 1]]).astype("double")

    signatory.signature(torchX, 10)

