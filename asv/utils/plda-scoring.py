#coding:utf-8
import numpy as np
import sys
import os
from sklearn import metrics


# Copyright xmuspeech （Author:JFZhou,Fuchuan Tong 2021-1-5)



'''
s = φ^T_1Λφ_2 + φ^T_2 Λφ_1 + φ^T_1Γφ_1 + φ^T_2Γφ_2 +(φ_1 + φ_2)T_c + k
where
Γ = −1/4(Σ_{wc} + 2Σ_{ac})^{−1} − 1/4Σ^{−1}_{wc} + 1/2Σ^{−1}_{tot}
Λ = −1/4(Σ_{wc} +2Σ_{ac})^{−1} + 1/4Σ^{−1}_{wc}
c = (( Σ_{wc} +2Σ_{ac})^{−1} −Σ^{−1}_{tot})μ
k = log|Σtot|−1/2log|Σ_{wc} +2Σ_{ac}|−1/2log|Σ_{wc}|+μ^T(Σ^{−1}_{tot} −(Σ_{wc} +2Σ_{ac})^{−1})μ. 
'''

def PLDAScoring(enroll_xvector,test_xvector,Gamma,Lambda,c,k):

    score = np.matmul(np.matmul(enroll_xvector.T,Lambda),test_xvector) + np.matmul(np.matmul(test_xvector.T,Lambda),enroll_xvector) \
        + np.matmul(np.matmul(enroll_xvector.T,Gamma),enroll_xvector) + np.matmul(np.matmul(test_xvector.T,Gamma),test_xvector) \
        + np.matmul((enroll_xvector + test_xvector).T,c) + k
    
    return score[0][0]

def CalculateVar(between_var,within_var,mean):

    total_var_inv = np.linalg.inv(between_var + within_var)
    wc_add_2ac_inv = np.linalg.inv(within_var + 2* between_var)
    wc_inv = np.linalg.inv(within_var)

    # Gamma
    Gamma = (-1/4)*(wc_add_2ac_inv+wc_inv)+(1/2)*total_var_inv

    #Lambda
    Lambda = (-1/4)*(wc_add_2ac_inv-wc_inv)

    #c
    c = np.matmul((wc_add_2ac_inv-total_var_inv),mean)
    
    # Since k is a constant for the addition of all scores and does not affect the eer value, it is not counted as a scoring term
    # k = logdet_tot -(1/2) *(logdet_w_two_b+logdet_w) + np.matmul(np.matmul(mean.T,np.linalg.inv(total_var)-np.linalg.inv(within_var + 2* between_var)),mean)
    k = 0

    return Gamma,Lambda,c,k

def read_vec(xvector_file):
    spk2vectors = {}
    with open(xvector_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            key = line[0]
            vector = np.array([float(x) for x in line[1:]])
            spk2vectors[key] = vector
    return spk2vectors

def main(plda,enroll_xvector,test_xvector,trials,scores):

    with open(plda,'r') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == 'mean':
                mean = np.array([float(x) for x in line[1:]]).reshape(-1,1)
                dim = len(line[1:])
            elif line[0] == 'within_var':
                within_var = np.array([float(x) for x in line[1:]]).reshape(dim, dim)
            else:
                between_var = np.array( [float(x) for x in line[1:]] ).reshape(dim, dim)

    within_var = within_var + 5e-5*np.eye(within_var.shape[0])

    Gamma,Lambda,c,k = CalculateVar(between_var,within_var,mean)

    f_writer = open(scores,'w')
    enrollutt2vector = read_vec(enroll_xvector)
    testutt2vector = read_vec(test_xvector)

    labels = []
    scores = []
    with open(trials,'r') as f:
        for line in f:
            label, enroll, test= line.strip().split()
            labels.append(int(label))
            score = PLDAScoring(enrollutt2vector[enroll].reshape(-1,1),testutt2vector[test].reshape(-1,1),\
                                                                Gamma,Lambda,c,k)
            scores.append(score)
            f_writer.write(enroll+' '+test+ ' ' + label + ' '+str(score) +'\n')
    f_writer.close()
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    print('EER=%f , thresholds=%f' % (eer, thresholds[idxE]))


if __name__ == "__main__":
  
    if len(sys.argv) != 6:
        print("Usage: "+sys.argv[0]+" <plda> <enroll-xvector-rspecifier> <test-xvector-rspecifier> <trials-rxfilename> <scores-wxfilename>")
        print("e.g.: "+sys.argv[0]+" --num-utts=ark:exp/enroll/num_utts.ark plda ark:exp/enroll/spk_xvectors.ark ark:exp/test/xvectors.ark trials scores")
        exit(1)

    plda = sys.argv[1]
    enroll_xvector = sys.argv[2]
    test_xvector = sys.argv[3]
    trials = sys.argv[4]
    scores = sys.argv[5]

    main(plda,enroll_xvector,test_xvector,trials,scores)
