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
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from asv.dataset.dataset import AudioDataset, CollateFunc
from asv.utils.checkpoint import load_checkpoint
from asv.nnet.ecapa import ECAPA_TDNN, AAMsoftmax, SVModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--result_file', required=True, help='asr result file')
    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    test_collate_conf = configs['collate_conf']
    test_collate_conf['wav_aug'] = False
    test_collate_func = CollateFunc(**test_collate_conf)
    test_dataset = AudioDataset(args.test_data, batch_size=1)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=10)

    output_dim = configs['output_dim']
    linear_units = configs['linear_units']
    ecapatdnn = ECAPA_TDNN(linear_units)
    ecapaloss = AAMsoftmax(output_dim)
    model = SVModel(ecapa_tdnn=ecapatdnn, aam_softmax=ecapaloss)
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    ecapa = model.speaker_encoder
    ecapa.eval()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, labels= batch
            feats = feats.to(device)
            emd = ecapa(feats)
            emd = torch.nn.functional.normalize(emd, p=2, dim=1)
            for i, key in enumerate(keys):
                key = keys[i]
                if use_cuda:
                    embedding = ' '.join(['{:.6f}'.format(x) for x in emd[i].cpu().numpy().tolist()])
                else:
                    embedding = ' '.join(['{:.6f}'.format(x) for x in emd[i].numpy().tolist()])
                logging.info('{}'.format(key))
                fout.write('{} {}\n'.format(key, embedding))
