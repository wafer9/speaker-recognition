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

    parser.add_argument("--top-n", type=int, default=300,
                        help="Used in AS-Norm.")
    parser.add_argument("--cross-select", type=str, default="false", choices=["true", "false"],
                        help="Used in AS-Norm. "
                             "If true, select top n enroll/test keys by test/enroll_cohort scores. "
                             "If false, select top n enroll/test keys by enroll/test_cohort scores.")
    parser.add_argument("--enroll_test_same", type=str, default="false", choices=["true", "false"], help=" ")

    parser.add_argument("--trials", type=str, help="trials")
    parser.add_argument("--imposter_cohort", type=str, help="Original embedding for imposter.")
    parser.add_argument("--enroll", type=str, help="Original embedding for enroll, test")
    parser.add_argument("--test", type=str, help="Original embedding for enroll, test")

    args = parser.parse_args()

    return args


def load_score(emd_path):
    logging.info("Load embedding from {} ...".format(emd_path))
    embs = {}
    with open(emd_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            embs[line[0]] = [float(x) for x in line[1:]]
    return embs


def cosine(e, t):
    cos = 0
    for i in range(len(e)):
        cos += e[i] * t[i]
    return cos


def cos_score(enroll_test, imposter):
    res = []
    logging.info("compute cos score ...")
    imposter_key = []
    imposter_emd = []
    for key in imposter.keys():
        imposter_key.append(key)
        imposter_emd.append(imposter[key])
    imposter_emd = torch.Tensor(imposter_emd)
    for key1 in tqdm.tqdm(enroll_test.keys(), total=len(enroll_test)):
        e = torch.Tensor(enroll_test[key1]).unsqueeze(1)
        score = torch.mm(imposter_emd, e).squeeze(1)
        for i in range(len(imposter_key)):
            res.append([key1, imposter_key[i], score[i].item()])
    data = DataFrame(res)
    data.rename(columns={0: 'enroll', 1: 'imposter', 2: 'score'}, inplace=True)
    return data


def snorm(args):
    """ Symmetrical Normalization.
    Reference: Kenny, P. (2010). Bayesian speaker verification with heavy-tailed priors. Paper presented at the Odyssey.
    """

    imposter = load_score(args.cohort)
    enroll = load_score(args.enroll)
    test = load_score(args.test)

    enroll_cohort_score = cos_score(enroll, imposter)
    test_cohort_score = cos_score(test, imposter)

    output_score = []

    logging.info("Use Symmetrical Normalization (S-Norm) to normalize scores ...")

    # This .groupby function is really an efficient method than 'for' grammar.
    enroll_group = enroll_cohort_score.groupby("enroll")
    test_group = test_cohort_score.groupby("enroll")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    f = open(args.trials)
    for line in f:
        enroll_key, test_key, score = line.strip().split()
        normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] +
                              (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    logging.info("Normalize scores done.")


def asnorm(args):
    """ Adaptive Symmetrical Normalization.
    Reference: Cumani, S., Batzu, P. D., Colibro, D., Vair, C., Laface, P., & Vasilakakis, V. (2011). Comparison of
               speaker recognition approaches for real applications. Paper presented at the Twelfth Annual Conference
               of the International Speech Communication Association.

               Cai, Danwei, et al. “The DKU-SMIIP System for NIST 2018 Speaker Recognition Evaluation.” Interspeech 2019,
               2019, pp. 4370–4374.

    Recommend: Matejka, P., Novotný, O., Plchot, O., Burget, L., Sánchez, M. D., & Cernocký, J. (2017). Analysis of
               Score Normalization in Multilingual Speaker Recognition. Paper presented at the Interspeech.

    """

    imposter = load_score(args.imposter_cohort)
    enroll = load_score(args.enroll)
    test = load_score(args.test)

    enroll_cohort_score = cos_score(enroll, imposter)
    if args.enroll_test_same == "true":
        test_cohort_score = enroll_cohort_score
    else:
        test_cohort_score = cos_score(test, imposter)

    logging.info("Use Adaptive Symmetrical Normalization (AS-Norm) to normalize scores ...")

    # Note that, .sort_values function will return NoneType with inplace=True and .head function will return a DataFrame object.
    # The order sort->groupby is equal to groupby->sort, so there is no problem about independence of trials.
    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)

    enroll_group = enroll_cohort_score.groupby("enroll").head(args.top_n).groupby("enroll")
    test_group = test_cohort_score.groupby("enroll").head(args.top_n).groupby("enroll")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    labels = []
    scores = []
    f = open(args.trials)
    for line in f:
        label, enroll_key, test_key = line.strip().split()
        score = cosine(enroll[enroll_key], test[test_key])
        if args.cross_select == "true":
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] +
                                  (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] +
                                  (score - test_mean[test_key]) / test_std[test_key])
        labels.append(int(label))
        scores.append(normed_score)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100
    print('EER=%f , thresholds=%f' % (eer, thresholds[idxE]))


def main():
    print(" ".join(sys.argv))
    args = get_args()
    try:
        if args.method == "snorm":
            snorm(args)
        elif args.method == "asnorm":
            asnorm(args)
        else:
            raise TypeError("Do not support {} score normalization.".format(args.method))
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



