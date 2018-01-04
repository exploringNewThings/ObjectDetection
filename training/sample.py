import argparse
import logging
import os
import torch
import torch.distributed as dist
from networks.inception import inception_v3
from networks.inception import Inception3Conv
import numpy as np
import logging

def main():
    model = inception_v3(pretrained=True)
    
    randdata = torch.autograd.Variable(torch.randn(1,3,299,299)).cuda()
    model.cuda()
    
    out = model.forward(randdata)
    print(out)

if __name__ == '__main__':
    main()
