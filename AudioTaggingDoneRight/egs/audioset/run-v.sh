#!/bin/bash

export TORCH_HOME=../../pretrained_models
hostname=$(hostname | cut -d'.' -f1)
echo "Current Machine is ${hostname}"
slurm_id=12913345
#$(squeue -u billyli | grep $hostname | awk '{ print $1; }')
echo "Slurm ID is __${slurm_id}__"
python prep_AudioSet.py --slurm-id ${slurm_id}
slurm_folder=/local/slurm-${slurm_id}/local/audio/data/datafiles
model=mmt
dataset=audioset_s
# full or balanced for audioset
set=balanced
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  tr_data=${slurm_folder}/audioset_bal_train_data_v.json
else
  bal=bal
  lr=1e-5
  epoch=10
  tr_data=${slurm_folder}/audioset_bal_unbal_train_data_v.json
fi
te_data=${slurm_folder}/audioset_eval_data_v.json
freqm=12
n_mels=64
timem=60
mixup=0.3
# corresponding to overlap of 6 for 16*16 patches
fstride=4
tstride=4
batch_size=24
mean=-29.686901
std=40.898224
suffix=ast_super_res
exp_dir=./exp/lauratest7-${model}-${dataset}-${set}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-fm${freqm}-tm${timem}-mix${mixup}-m${mean}-std${std}-epoch${epoch}-${suffix}
expid=-${model}-${set}-f${fstride}-t${tstride}-pre${imagenetpretrain}-b${batch_size}-lr${lr}-mix${mixup}-freqm${freqm}-timem${timem}-m${mean}-std${std}
logger=${exp_dir}/${expid}.txt
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir
echo ${model}
echo "CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels}\
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir --mean ${mean} --std ${std} \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain" >> $logger
CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir --mean ${mean} --std ${std} \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain
