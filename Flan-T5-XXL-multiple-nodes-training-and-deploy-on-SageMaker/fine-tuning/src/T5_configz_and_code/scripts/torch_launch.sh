#!/bin/bash

#Please change the folder "T5_configz_and_code" to your folder which includes config files and main codes. 
WORKING_DIR=/opt/ml/code/T5_configz_and_code
SM_WORKING_DIR=/opt/ml/model

#The related information about multi-nodes cluster.
MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"

#Configure the distributed arguments for torch.distributed.launch.
GPUS_PER_NODE="$SM_NUM_GPUS"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"

#Set the path of your deepspeed config file.
DS_CONFIG="${WORKING_DIR}/configs/ds_flan_t5_z3_config_bf16.json"

#Configure the parameters for your training according to your model and dataset.
#Note: you should set the corresponding paths of train_dataset_path and test_dataset_path according to your input data channel name.
EPOCHS=1
model_id="google/flan-t5-xxl"
train_dataset_path='/opt/ml/input/data/training'
test_dataset_path='/opt/ml/input/data/test'
learning_rate=0.0001
generation_max_length=150
per_device_train_batch_size=1
per_device_eval_batch_size=8

OPTS=""
OPTS+=" --per_device_eval_batch_size ${per_device_eval_batch_size}"
OPTS+=" --per_device_train_batch_size ${per_device_train_batch_size}"
OPTS+=" --generation_max_length ${generation_max_length}"
OPTS+=" --test_dataset_path ${test_dataset_path}"
OPTS+=" --model_id ${model_id}"
OPTS+=" --train_dataset_path ${train_dataset_path}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --learning_rate ${learning_rate}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --epochs ${EPOCHS}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/scripts/run_seq2seq_deepspeed.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
