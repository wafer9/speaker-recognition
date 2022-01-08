# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
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

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from asv.dataset.dataset import AudioDataset, CollateFunc
from asv.nnet.ecapa import ECAPA_TDNN, AAMsoftmax, SVModel
from asv.utils.checkpoint import load_checkpoint, save_checkpoint
from asv.utils.executor import Executor
from asv.utils.scheduler import WarmupLR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--veri_test', required=True, help='verification test file')
    parser.add_argument('--musan_file', required=True, help='musan noise data file')
    parser.add_argument('--rir_file', required=True, help='rir data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)
    print(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    configs['collate_conf']['musan_file'] = args.musan_file
    configs['collate_conf']['rir_file'] = args.rir_file
    configs['veri_test'] = args.veri_test

    distributed = args.world_size > 1

    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = AudioDataset(args.train_data, **dataset_conf)
    cv_dataset = AudioDataset(args.cv_data, batch_size=1)

    train_collate_func = CollateFunc(**configs['collate_conf'])
    cv_collate_conf = copy.deepcopy(configs['collate_conf'])
    cv_collate_conf['wav_aug'] = False
    cv_collate_func = CollateFunc(**cv_collate_conf)

    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=train_collate_func,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   pin_memory=args.pin_memory,
                                   batch_size=1,
                                   num_workers=args.num_workers,
                                   drop_last = True)
    cv_data_loader = DataLoader(cv_dataset,
                                   collate_fn=cv_collate_func,
                                   shuffle=False,
                                   batch_size=1,
                                   num_workers=args.num_workers)
    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = configs['collate_conf']['feature_extraction_conf']['mel_bins']
    configs['output_dim'] = train_dataset.output_dim
    print(configs['output_dim'])
    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init sv model from configs
    ecapatdnn = ECAPA_TDNN(configs['linear_units'])
    ecapaloss = AAMsoftmax(configs['output_dim'])
    model = SVModel(ecapa_tdnn=ecapatdnn, aam_softmax=ecapaloss)
    if args.rank == 0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    #if args.rank == 0:
    #    script_model = torch.jit.script(model)
    #    script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False)
        device = torch.device("cuda")
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)
    gamma = configs['optim_conf']['gamma']
    lr =  configs['optim_conf']['lr'] * gamma ** start_epoch
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=gamma)

    final_epoch = None
    configs['rank'] = args.rank
    configs['is_distributed'] = distributed
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        executor.train(model, optimizer, 
                       scheduler, train_data_loader, device,
                       writer, configs, scaler=None)
        if args.rank == 0:
            EER, minDCF = executor.cv(model, cv_data_loader, device, configs)
            log_str = 'CV {} EER {:.6f}% '.format( epoch, EER)
            logging.debug(log_str)            
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'EER': float(EER),
                })
            writer.add_scalar('epoch/eer', EER, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()

