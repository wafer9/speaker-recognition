#!/bin/bash
. ./path.sh || exit 1;
# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=3 # start from 0 if you need to start from data preparation
stop_stage=3
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0
# data

nj=16

train_set=train
train_config=conf/train_ecapa.yaml
dir=exp/ecapa_v1_v2
checkpoint=${dir}/89.pt

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/75.pt

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    # You had better rm it manually before you start run.sh on first node.
    # rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    # The number of gpus runing on each node/machine
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="gloo"
    # The total number of processes/gpus, so that the master knows
    # how many workers to wait for.
    # More details about ddp can be found in
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    world_size=`expr $num_gpus \* $num_nodes`
    echo "total gpus is: $world_size"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        # Rank of each gpu/process used for knowing whether it is
        # the master of a worker.
        rank=`expr $node_rank \* $num_gpus + $i`
        python asv/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --train_data data/voxceleb1_2/train.list \
            --cv_data data/voxceleb1_2/test.list \
            --veri_test data/veri_test2.txt \
            --musan_file data/musan.list \
            --rir_file data/rir.list \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 10 \
            --pin_memory
    } &
    done
    wait
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p ${dir}/voxceleb2
    test_data="data/voxceleb2/train.list"
    num=$(cat $test_data | wc -l)
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    lines=`expr $num \/ $num_gpus + 1`
    split -a 1 -d -l $lines ${test_data} ${test_data}_
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        python asv/bin/extract.py --gpu $gpu_id \
            --config $dir/train.yaml \
            --test_data ${test_data}_$gpu_id \
            --checkpoint $decode_checkpoint \
            --result_file ${dir}/voxceleb2/train_$gpu_id.emd 
    } &
    done
    wait
    cat ${dir}/voxceleb2/train_$gpu_id.emd >${dir}/voxceleb2/train.emd 
    
    mkdir -p ${dir}/test
    for part in voxceleb1 june july; do
        python asv/bin/extract.py --gpu 0  \
            --config $dir/train.yaml \
            --test_data data/test/${part}.list \
            --checkpoint $decode_checkpoint \
            --result_file ${dir}/test/${part}.emd
    done
fi

# PLDA
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python asv/utils/plda-train.py \
        ${dir}/voxceleb2/train.emd \
        ${dir}/voxceleb2/plda

    python asv/utils/plda-scoring.py  ${dir}/voxceleb2/plda.ori \
        ${dir}/test/voxceleb1.emd  ${dir}/test/voxceleb1.emd \
        data/veri_test2.txt  ${dir}/test/voxceleb1.score
    python asv/utils/plda-scoring.py  ${dir}/voxceleb2/plda.ori \
        ${dir}/test/july.emd  ${dir}/test/july.emd \
        data/zyb/veri_test1.txt  ${dir}/test/july_july.score
    python asv/utils/plda-scoring.py  ${dir}/voxceleb2/plda.ori \
        ${dir}/test/june.emd  ${dir}/test/july.emd \
        data/zyb/veri_test2.txt  ${dir}/test/june_july.score

fi

# PLDA
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python asv/utils/vec2ark.py ${dir}/voxceleb2/train.emd\
           ${dir}/voxceleb2/train.ark
    python asv/utils/vec2ark.py ${dir}/test/voxceleb1.emd \
           ${dir}/test/voxceleb1.ark
    ivector-mean ark:${dir}/voxceleb2/train.ark \
           ${dir}/voxceleb2/mean.vec


fi

# score normalization
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    imposter="${dir}/voxceleb2/train_10w.emd"
    cat ${dir}/voxceleb2/train.emd |shuf -n 100000 >$imposter
    python asv/utils/ScoreNormalization.py \
        --enroll_test_same true --imposter_cohort ${imposter} \
        --trials data/veri_test2.txt   \
        --enroll ${dir}/test/voxceleb1.emd\
        --test ${dir}/test/voxceleb1.emd

fi

