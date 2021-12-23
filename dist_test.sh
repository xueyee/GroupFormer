#!/bin/sh

work_path=$(dirname $(readlink -f $0))
currenttime=`date "+%Y%m%d%H%M%S"`

config=$3
g=$(($1<16?$1:16))

declare projname
projname=`basename ${config} .yaml`


if [ ! -d "${work_path}/experiments/${projname}/log" ]; then
mkdir -p "${work_path}/experiments/${projname}/log"
fi


echo 'GPU num = ' $1


PY_ARGS=${@:4}

python -u ./main.py \
--config=$2 \
--checkpoint=$3 \
--evaluate \
${PY_ARGS} \
2>&1 | tee ${work_path}/experiments/${projname}/log/test_${projname}_${currenttime}_$1'gpu'.log
