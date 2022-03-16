# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
# Copyright (c) 2021 Jinsong Pan
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

import argparse
import codecs
import copy
import logging
import random

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.sox_effects as sox_effects
import yaml
from PIL import Image
from PIL.Image import BICUBIC
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from asv.dataset.wav_distortion import distort_wav_conf

torchaudio.set_audio_backend("sox_io")
from scipy import signal
import soundfile
import random

class CollateFunc(object):
    """ Collate function for AudioDataset
    """
    def __init__(
        self,
        musan_file,
        rir_file,
        feature_extraction_conf=None,
        wav_aug=True,
        wav_aug_conf=None,
    ):
        """
        Args:
            input is raw wav and feature extraction is needed.
        """
        self.feature_extraction_conf = feature_extraction_conf
        self.wav_aug = wav_aug
        self.wav_aug_conf = wav_aug_conf
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        if self.wav_aug:
            self.length = self.wav_aug_conf['length']
            self.add_rev = self.wav_aug_conf['add_rev']
            self.add_noise = self.wav_aug_conf['add_noise']
        augment_files = open(musan_file).read().splitlines()
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = open(rir_file).read().splitlines()

    def __call__(self, batch):
        assert (len(batch) == 1)
        if self.wav_aug:
            self.length = random.randint(240, 1600)/1000.0
        keys, xs, labels = self._extract_feature(batch[0], self.feature_extraction_conf)
        xs = torch.Tensor(xs)
        xs = xs - torch.mean(xs, dim=1, keepdim=True)
        return keys, xs, torch.LongTensor(labels)

    def _extract_feature(self, batch, feature_extraction_conf):
        """ Extract acoustic fbank feature from origin waveform.

        Speed perturbation and wave amplitude distortion is optional.

        Args:
            batch: a list of tuple (wav id , wave path).
            feature_extraction_conf:a dict , the config of fbank extraction.
        Returns:
            (keys, feats, labels)
        """
        keys = []
        feats = []
        labels = []
        for i, x in enumerate(batch):
            try:
                wav_path = x[2]
                waveform, sr = torchaudio.load(wav_path)
                if self.wav_aug:
                    waveform = waveform[0]
                    length = int(self.length * sr + 240)
                    if waveform.shape[0] <= length:
                        shortage = length - waveform.shape[0]
                        waveform = np.pad(waveform, (0, shortage), 'wrap')
                    start_frame = np.int64(random.random() * (waveform.shape[0] - length))
                    waveform = waveform[start_frame:start_frame + length]
                    waveform = np.stack([waveform], axis=0)
                    augtype = random.randint(0, 5)
                    if augtype == 0:   # Original
                        waveform = waveform
                    elif augtype == 1 and self.add_rev: # Reverberation
                        waveform = self._add_rev(waveform)
                    elif augtype == 2 and self.add_noise: # Babble
                        waveform = self._add_noise(waveform, 'speech')
                    elif augtype == 3 and self.add_noise: # Music
                        waveform = self._add_noise(waveform, 'music')
                    elif augtype == 4 and self.add_noise: # Noise
                        waveform = self._add_noise(waveform, 'noise')
                    elif augtype == 5 and self.add_noise: # Television noise
                        waveform = self._add_noise(waveform, 'speech')
                        waveform = self._add_noise(waveform, 'music')
                else:
                    waveform = waveform.numpy()
                waveform = waveform * (1 << 15)
                waveform = torch.from_numpy(waveform)
                if feature_extraction_conf['feature_type'] == 'fbank':
                    mat = kaldi.fbank(
                        waveform,
                        num_mel_bins=feature_extraction_conf['mel_bins'],
                        frame_length=feature_extraction_conf['frame_length'],
                        frame_shift=feature_extraction_conf['frame_shift'],
                        dither=feature_extraction_conf['wav_dither'],
                        energy_floor=0.0,
                        sample_frequency=sr)
                elif feature_extraction_conf['feature_type'] == 'mfcc':
                    mat = kaldi.mfcc(
                        waveform,
                        num_ceps=feature_extraction_conf['mel_bins'],
                        num_mel_bins=feature_extraction_conf['mel_bins'],
                        frame_length=feature_extraction_conf['frame_length'],
                        frame_shift=feature_extraction_conf['frame_shift'],
                        dither=feature_extraction_conf['wav_dither'],
                        energy_floor=0.0,
                        sample_frequency=sr)

                mat = mat.detach().numpy()
                feats.append(mat)
                keys.append(x[0])
                labels.append(x[1])
            except (Exception) as e:
                print(e)
                logging.warn('read utterance {} error'.format(x[0]))
                pass
        return keys, feats, labels

    def _add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = torchaudio.load(rir_file)
        rir = rir.numpy()
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :int(self.length * sr) + 240]

    def _add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = int(self.length * sr + 240)
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class AudioDataset(Dataset):
    def __init__(self,
                 data_file,
                 batch_size=1,):
        """Dataset for loading audio data.
        """
        data = []
        labels = []
        # Open in utf8 mode since meet encoding problem
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split(' ')
                if len(arr) != 3:
                    continue
                data.append((arr[0], arr[1], arr[2]))
                labels.append(int(arr[1]))
        labels = list(set(labels))
        labels.sort()
        labels_dict = {target : ii for ii, target in enumerate(labels)}
        self.minibatch = []
        num_data = len(data)
        cur = 0
        while cur < num_data:
            end = min(cur + batch_size, num_data)
            if batch_size > 1 and end != cur + batch_size:
                break
            item = []
            for i in range(cur, end):
                speaker_label = labels_dict[int(data[i][1])]
                item.append((data[i][0], speaker_label, data[i][2]))
            self.minibatch.append(item)
            cur = end
        self.output_dim = len(labels_dict)

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, idx):
        return self.minibatch[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='config file')
    parser.add_argument('--data_file', help='input data file')
    parser.add_argument('--musan_file', help='input musan file')
    parser.add_argument('--rir_file', help='input rir file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Init dataset and data loader
    collate_conf = copy.copy(configs['collate_conf'])
    collate_conf['musan_file'] = args.musan_file
    collate_conf['rir_file'] = args.rir_file
    collate_func = CollateFunc(**collate_conf)
    dataset_conf = configs.get('dataset_conf', {})
    dataset = AudioDataset(args.data_file, **dataset_conf)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             sampler=None,
                             num_workers=0,
                             collate_fn=collate_func)
    for i, batch in enumerate(data_loader):
        print(i)
        # print(batch[1].shape)

