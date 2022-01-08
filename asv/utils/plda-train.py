# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2019-12-22)

import scipy
import numpy as np
import math
import os
import sys
from plda_base import PldaStats,PldaEstimation
import logging


# Logger
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():

    if len(sys.argv)!=3:
  
        print("Usage: "+sys.argv[0]+" <ivector-rspecifier> <plda>\n")
        print("e.g.: "+sys.argv[0]+" vectors.ark plda")
        
        sys.exit() 

    ivectors_reader = sys.argv[1]
    plda_out = sys.argv[2]


    logger.info('Load vecs and accumulate the stats of vecs.....')
    utt2spk_dict = {}
    with open(ivectors_reader,'r') as f:
        for line in f:
            utt = line.strip().split()[0]
            spk = utt.split('-')[0]
            utt2spk_dict[utt] = spk

    spk2vectors = {}
    with open(ivectors_reader, 'r') as f:
        for line in f:
            line = line.strip().split()
            key = line[0]
            vector = [float(x) for x in line[1:]]
            dim = len(vector)
            spk = utt2spk_dict[key]
            if key not in spk2vectors:
                spk2vectors[spk] = [vector]
            else:
                spk2vectors[spk].append(vector)


    plda_stats=PldaStats(dim)
    for key in spk2vectors.keys():
        vectors = np.array(spk2vectors[key], dtype=float)
        weight = 1.0
        plda_stats.add_samples(weight,vectors)

    logger.info('Estimate the parameters of PLDA by EM algorithm...')
    plda_stats.sort()
    plda_estimator=PldaEstimation(plda_stats)
    plda_estimator.estimate(num_em_iters=1)
    logger.info('Save the parameters for the PLDA adaptation...')
    plda_estimator.plda_write(plda_out+'.ori')
    plda_trans = plda_estimator.get_output()
    logger.info('Save the parameters for scoring directly, which is the same with the plda in kaldi...')
    plda_trans.plda_trans_write(plda_out)

if __name__ == "__main__":
    main()