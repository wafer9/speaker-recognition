# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-26 2020-05-26)

import sys
import logging
import argparse
import traceback
from pandas.core.frame import DataFrame
from sklearn import metrics
import numpy, tqdm
import logging
import torch
import multiprocessing

# Parse
def get_args():
    """Score:
            <key1, key2, score>
    Recommend: subset 2000 ~ 3000 utts from trainset as a cohort set and use asnorm with top-n=200 ~ 400.
    """
    parser = argparse.ArgumentParser(
        description="Score Normalization.")

    parser.add_argument("--method", default="asnorm", type=str,
                        choices=["snorm", "asnorm"],
                        help="Choices to select a score normalization.")

    parser.add_argument("--top-n", type=int, default=300, help="Used in AS-Norm.")
    parser.add_argument("--enroll_test_same", type=str, default="false", choices=["true", "false"], help=" ")

    parser.add_argument("--trials", type=str, help="trials")
    parser.add_argument("--imposter_cohort", type=str, help="Original embedding for imposter.")
    parser.add_argument("--enroll", type=str, help="Original embedding for enroll, test")
    parser.add_argument("--test", type=str, help="Original embedding for enroll, test")

    args = parser.parse_args()

    return args


def load_score(emd_path):
    logging.info("Load embedding from {} ...".format(emd_path))
    keys = []
    embs = {}
    with open(emd_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            keys.append(line[0])
            embs[line[0]] = [float(x) for x in line[1:]]
    return keys, embs


def cosine(e, t):
    cos = 0
    for i in range(len(e)):
        cos += e[i] * t[i]
    return cos


def cos_score(keys, enroll_test, imposter, k):
    res = []
    logging.info("compute cos score ...")
    imposter_key = []
    imposter_emd = []
    for key in imposter.keys():
        imposter_key.append(key)
        imposter_emd.append(imposter[key])
    imposter_emd = torch.Tensor(imposter_emd)

    for key in tqdm.tqdm(keys, total=len(keys)):
        e = torch.Tensor(enroll_test[key]).unsqueeze(1)
        score = torch.mm(imposter_emd, e).squeeze(1)
        score_e_c = torch.topk(score, k=k)[0]
        mean_e_c = torch.mean(score_e_c, dim=0)
        std_e_c = torch.std(score_e_c, dim=0)
        res.append([key, mean_e_c, std_e_c])
    return res



def main():
    print(" ".join(sys.argv))
    args = get_args()
    key_i, imposter = load_score(args.imposter_cohort)
    key_e, enroll = load_score(args.enroll)
    key_t, test = load_score(args.test)

    nj = 10
    pool = multiprocessing.Pool(processes=nj) 
    num = int(len(key_e)/nj) + 1
    results = []
    for i in range(nj):
        start = num * i
        end = min(num * i + num, len(key_e))
        keys = key_e[start:end]
        r = pool.apply_async(cos_score, (keys, enroll, imposter, args.top_n))
        results.append(r)
        logging.info("process {} start".format(i))
    pool.close()
    pool.join()
    enroll_c = {}
    for i in results:
        for j in i.get():
            enroll_c[j[0]] = [j[1], j[2]]
    if args.enroll_test_same == 'true':
        test_c = enroll_c
    else:
        pool = multiprocessing.Pool(processes=nj)
        num = int(len(key_t)/nj) + 1
        results = []
        for i in range(nj):
            start = num * i
            end = min(num * i + num, len(key_t))
            keys = key_t[start:end]
            r = pool.apply_async(cos_score, (keys, test, imposter, args.top_n))
            results.append(r)
            logging.info("process {} start".format(i))
        pool.close()
        pool.join()
        test_c = {}
        for i in results:
            for j in i.get():
                test_c[j[0]] = [j[1], j[2]]

    logging.info("Use Adaptive Symmetrical Normalization (AS-Norm) to normalize scores ...")

    labels = []
    scores = []
    f = open(args.trials)
    for line in f:
        label, enroll_key, test_key = line.strip().split()
        score = cosine(enroll[enroll_key], test[test_key])
        normed_score = 0.5 * ((score - enroll_c[enroll_key][0]) / enroll_c[enroll_key][1] +
                              (score - test_c[test_key][0]) / test_c[test_key][1])
        labels.append(int(label))
        scores.append(normed_score)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100
    print('EER=%f , thresholds=%f' % (eer, thresholds[idxE]))
 


if __name__ == "__main__":
    main()



