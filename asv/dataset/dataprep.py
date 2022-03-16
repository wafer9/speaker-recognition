#!/usr/bin/python

import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile

parser = argparse.ArgumentParser(description = "VoxCeleb downloader");

parser.add_argument('--audio_path',     type=str, default="data", help='audio directory');
#parser.add_argument('--list',     type=str, default="data", help='Target list');
args = parser.parse_args();

files = glob.glob('%s/*/*.flac'%args.audio_path)
files.sort()
for fname in tqdm(files):
    name = fname.split("/")[-2] + "-" + fname.split("/")[-1].split(".")[0]
    spk = int(fname.split("/")[-2].split("id")[1]) + 10000
    print(name, spk, fname)
