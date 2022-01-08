import kaldi_io
import numpy
import sys

fin = sys.argv[1]
fout = sys.argv[2]

vecs = {}
with open(fin, 'r') as f:
    for line in f:
        line = line.strip().split()
        key = line[0]
        vec = numpy.array([float(x) for x in line[1:]])
        vecs[key] = vec

with open(fout,'wb') as f:
    for key, vec in vecs.items():
        kaldi_io.write_vec_flt(f, vec, key=key)
