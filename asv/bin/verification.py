# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import logging
import torch
from sklearn import metrics
import numpy

def cosine(e, t):
    cos = 0
    for i in range(len(e)):
        cos += e[i] * t[i]
    return cos

def load_score(emd_path):
    logging.info("Load embedding from {} ...".format(emd_path))
    embs = {}
    with open(emd_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            embs[line[0]] = [float(x) for x in line[1:]]
    return embs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--trials', required=True, help='config file')
    parser.add_argument('--enroll', required=True, help='test data file')
    parser.add_argument('--test', required=True, help='checkpoint model')
    args = parser.parse_args()
    print(args)

    enroll = load_score(args.enroll)
    test = load_score(args.test)

    labels = []
    scores = []
    with open(args.trials, 'r') as f:
        for line in f:
            line = line.strip().split()
            labels.append(int(line[0]))
            e = enroll[line[1]]
            t = test[line[2]]
            scores.append(cosine(e, t))

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100
    print('EER=%f , thresholds=%f' % (eer, thresholds[idxE]))


