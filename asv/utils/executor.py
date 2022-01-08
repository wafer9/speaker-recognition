# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch, numpy
from torch.nn.utils import clip_grad_norm_
from sklearn import metrics
from operator import itemgetter

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):	
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)

        num_seen_utts = 0
        total_loss, total_acc = 0, 0
        num_total_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            keys, feats, labels = batch
            context = None
            if is_distributed and batch_idx % accum_grad != 0 :
                context = model.no_sync
            else:
                context = nullcontext
            with context():
                loss, acc = model(feats, labels)
                loss = loss / accum_grad
                loss.backward()
            if batch_idx % accum_grad == 0:
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                optimizer.zero_grad()
            if torch.isfinite(loss):
                num_utts = len(keys)
                num_seen_utts += num_utts
                total_loss += loss.item() * accum_grad * num_utts
                total_acc += acc.item() * num_utts
            if batch_idx % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                log_str = 'TRAIN Batch {}/{} loss {:.6f} acc {:.6f} '.format(
                    batch_idx, num_total_batch, loss.item() * accum_grad, acc.item())
                log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                logging.debug(log_str)
        scheduler.step()
        total_loss /= num_seen_utts
        total_acc /= num_seen_utts
        logging.info('TRAIN info total_loss {:.6f} total_acc {:.6f} '.format(total_loss, total_acc))


    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        embeddings = {}
        logging.info('cv info num_utts {} '.format(len(data_loader)))
        for batch_idx, batch in enumerate(data_loader):
            keys, feats, labels = batch
            feats = feats.to(device)
            with torch.no_grad():
                emb = model.module.speaker_encoder(feats)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            for i in range(len(keys)):
                embeddings[keys[i]] = emb[i]
        scores, labels  = [], []
        lines = open(args['veri_test']).read().splitlines()
        for line in lines:
            embedding_1 = embeddings[line.split()[1]]
            embedding_2 = embeddings[line.split()[2]]
            # Compute the scores
            score = torch.mean(torch.matmul(embedding_1, embedding_2.T)) # higher is positive
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))
        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

