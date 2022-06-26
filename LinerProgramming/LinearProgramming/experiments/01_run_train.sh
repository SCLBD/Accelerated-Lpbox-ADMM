#!/bin/bash

# ---------- key factors -----------------
phase=train
net=mha
save_dir_policy=../saved_model/${net}/
# ----------------------------------------

batch_size=1024
iters_per_eval=25
L=3
rho=0.1
num_layer=20
num_output=20
var=0.14
lr=1e-4
dc=1e-4
mix='20-30-40'
subdir=snr-${mix}-L-${num_layer}-output-${num_output}-batch-${batch_size}-L-${L}-rho-${rho}-var-${var}-epo-${epoch}-itr-${iters_per_eval}-lr-${lr}-dc-${dc}
save_dir_model=../saved_model/lista/stage1/$subdir
model_dump=$save_dir_model/best_val_model.dump

p_iters_per_eval=3 #10
p_epoch=10  #
p_batch_size=256
dims='10-5'
classdims='64-64'
policy_type=sequential
kl=forward
p_lr=1e-3
post_dim=2 
sto=True

if [ ! -e $save_dir_policy ];
then
    mkdir -p $save_dir_policy
    mkdir -p $save_dir_policy/log
    mkdir -p $save_dir_policy/checkpoint
fi

python3 run.py \
    -kl_type ${kl} \
    -loss_type mle \
    -policy_type $policy_type \
    -gpu 0 \
    -batch_size $p_batch_size \
    -save_dir $save_dir_policy \
    -iters_per_eval $p_iters_per_eval \
    -learning_rate $p_lr \
    -weight_decay $dc \
    -num_epochs $p_epoch \
    -T_max $num_layer \
    -num_output $num_output \
    -val_model_dump $model_dump \
    -policy_hidden_dims $dims \
    -policy_multiclass_dims $classdims \
    -var $var \
    -phase $phase \
    -L $L \
    -rho $rho \
    -post_dim $post_dim \
    -snr_mix $mix \
    -net ${net} \
    -ws 100 
