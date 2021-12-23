#!/bin/sh

work_path=$(dirname $(readlink -f $0))
currenttime=`date "+%Y%m%d%H%M%S"`

config=$3
g=$(($2<16?$2:16))

declare projname
projname=`basename ${config} .yaml`


if [ ! -d "${work_path}/experiments/${projname}/log" ]; then
mkdir -p "${work_path}/experiments/${projname}/log"
fi


echo 'GPU num = ' $2

PY_ARGS=${@:4}

srun --mpi=pmi2 \
    --partition=$1 \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g \
    --job-name=${projname} \
    python -u ./main.py \
    --config=$3 \
    ${PY_ARGS} \
    2>&1 | tee ${work_path}/experiments/${projname}/log/train_${projname}_${currenttime}_$2'gpu'.log
