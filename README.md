#  Training LLM on Amazon SageMaker for multiple nodes with deepspeed

This repo will show the whole codes:
1. Fine tuning LLM by DeepSpeed on SageMaker for multiple nodes.
2. Deploy the trained model from above step #1 on SageMaker.

## Prerequisite:

a. Download the "s5cmd" command from source and uncompress it (using the following: curl -L https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz | tar -xz).

b. Clone this repo.

c. Move the "s5cmd" to the path: "Flan-T5-XXL-multiple-nodes-training-and-deploy-on-SageMaker/fine-tuning/src/T5_configz_and_code/scripts/".

The repo is tested successfully on Data Science image and Python 3 kernel of Sagemaker studio with ml.m5.large kernel gateway instance in us-east-1 region (If you encounter with kerenl restaring issue when preparing dataset in DeepSpeed-Flan-T5-on-Sagemaker-multiple-nodes.ipynb, I suggest that you shut down the kernel gateway instance and re-execute the DeepSpeed-Flan-T5-on-Sagemaker-multiple-nodes.ipynb).

## Fine tuning LLM such as Flan-T5-XXL

Now, we utilize the torch.distributed.launch + Deepspeed + Huggingface trainer API to fine tunig Flan-T5-XXL on AWS SageMaker for multiple nodes (Just set the environment variable "NODE_NUMBER" to 1, you can use the same codes for multiple GPUs training on single node). You can follow up the folder structure, and prepare your training script and configure related parameters in the torch_launch.sh script. If you also use the HF high level trainer API to train CausalLM (such as GPT-J) or Seq2seqLM (such as T5), there is very little code that needs to be modified.

I explain more about these files: start.py as user entry point will set some environment variables such as master's IP address and invoke the torch_launch.sh script. Most of parameters (including training parameters and torch distributed launcher parameters) should be configured in torch_launch.sh. Finally torch_launch.sh will invoke your training python script. Also, you can use the requirements.txt to install related python libraries.

Now, the codes uses the Huggingface/HF API to download the model assets form HF model hub. Maybe your SageMaker training job will encounter with the timeout issue when downloading model assets from HF model hub, just restart the SageMaker training job to re-try. Also, you can separatly downlaod the model assets from HF and directly upload them to Amazon S3 (do not tar and compress these files). Then in your training script, please use s5cmd to download them from S3 only on local rank 0 (use the torch.distributed.barrier() to sync up for every rank, please refer to https://github.com/yuhuiaws/finetuning-and-deploying-llama-on-Sagemaker/blob/main/finetuning-llama-by-deepspeed/train.py), it will speed up the model asset downloading compared with downloading them by use of HF API.

### Some useful tips:

1. There is the open source "s5cmd" file in this repo, we can use the "s5cmd" command to speedup the uploading model assets to S3 (do not tar and compress these model assets, just directly upload to S3) after saving model in the container's local path.
2. When using deepspeed zero stage 2 training LLM on muliple nodes in SageMaker, maybe it will hung untile the NCCL communication is timeout. When it happens, you can check the GPU memory utility of training instances from Amazon cloudwatch. In my experiment, the GPU memory utility is almost full (but OOM didn't occur), it may be a signal that you should switch to zero stage 3 (the issue disappears when I switch to zero 3).
3. By default, DeepSpeed expects that a multi-nodes environment uses a shared storage. If this is not the case and each node can only see the local filesystem，you need to set the parameter "save_on_each_node" of Seq2SeqTrainingArguments API or TrainingArguments API to true (in this repo, I didn't use share data store such as EFS to save model, so I set the "save_on_each_node" to true).
4. When using deepspeed to train on multiple GPUs, if the parameter "stage3_gather_16bit_weights_on_model_save" in deepseed config file is set to false, pytorch_modle.bin will not be generated in the end. You can use the zero_to_fp32.py script (it is located in the saved model assets path) to convert the deepspeed zero shared checkpoints to fp32 pytorch model bin on Sagemaker notebook instance or Sagemaker studio (the procedure will consume large memory and time). If the parameter "stage3_gather_16bit_weights_on_model_save" in deepseed config file is set to true, the pytorch_modle.bin will be generated in the end (stage3_gather_16bit_weights_on_model_save enables model fp16 weights consolidation when model gets saved. With large models and multiple GPUs this is an expensive operation both in terms of memory and speed). So how to configure the stage3_gather_16bit_weights_on_model_save parameter? It is up to you, and I will set it to true If the training speed does not drop significantly.  
5. When you use deepspeed multiple nodes training and set the parameter "load_best_model_at_end" (from Seq2SeqTrainingArguments or TrainingArguments API) to true, maybe error will happens when finishing training procedure. The error looks like the following: 

        Could not locate the best model at /tmp/checkpoint-60/pytorch_model.bin, if you are running 
        distributed training on multiple nodes, you should activate `--save_on_each_node`. 

  In fact, I have configured the parameter "save_on_each_node" to true (my environment: transformer 4.26.0，pytorch 1.10，python 3.8). I will only save best model, configure "load_best_model_at_end" to false and fix the issue.

6. If you just want to save the best model weights, you can set the parameter "output_dir" (from Seq2SeqTrainingArguments or TrainingArguments API) to temporary path such as '/tmp' on p4d.24xlarge ("/tmp" has the enough disk space to save); And if you want to save all of the checkpoint during the training, you can set the output_dir to the checkponit local path (it will impact the train speed for multi-nodes training. Because SageMaker will upload the checkpoint to S3 nearly real-time, it will occupy the networking bandwidth and impact the communication efficiency between nodes in the cluster).
7. When using parameter "compute_metrics" from Trainer or Seq2SeqTrainer API, the evaluation procedure is very slow. So if you just want to run successfully the whole training process, you can comment out the  "compute_metrics".
8. When your training script will download something from website (such as nltk.downlaod("punkt")), you should ensure only one process in the current node (local rank 0) downloaindg files, otherwise it may fail the training job. 

        Traceback (most recent call last):
          File "/opt/ml/code/T5_configz_and_code/scripts/run_seq2seq_deepspeed.py", line 26, in <module>
            nltk.download("punkt", quiet=True)
          File "/opt/conda/lib/python3.8/site-packages/nltk/downloader.py", line 777, in download
        for msg in self.incr_download(info_or_id, download_dir, force):
          File "/opt/conda/lib/python3.8/site-packages/nltk/downloader.py", line 642, in incr_download
        yield from self._download_package(info, download_dir, force)
          File "/opt/conda/lib/python3.8/site-packages/nltk/downloader.py", line 699, in _download_package
        os.makedirs(download_dir)
          File "/opt/conda/lib/python3.8/os.py", line 223, in makedirs
        mkdir(name, mode)
        FileExistsError: [Errno 17] File exists: '/root/nltk_data'
        [nltk_data] [Errno 2] No such file or directory:
        [nltk_data]     '/root/nltk_data/tokenizers/punkt.zip'
        [nltk_data] Error with downloaded zip file
        [nltk_data] [Errno 2] No such file or directory:
        [nltk_data]     '/root/nltk_data/tokenizers/punkt.zip'
        Downloading builder script:   0%|          | 0.00/6.27k [00:00<?, ?B/s]
        Downloading builder script: 100%|██████████| 6.27k/6.27k [00:00<00:00, 7.75MB/s]
        [nltk_data] Error with downloaded zip file
        [nltk_data] Error with downloaded zip file
        [nltk_data] Error with downloaded zip file

If you use the torch.distributed.launch, you can utilize the barrier function to achivement this purpose. More details, you can find the related code from run_seq2seq_deepspeed.py.

9. When you use torch.distributed.launch, please don't use global variables in your training script. Otherwise, the CUDA errors may occurs when exiting your training script and fails the training job. So in run_seq2seq_deepspeed.py, I change the "metric" variable from global variable to local variable.
10. Plesae do not save the model into "/opt/ml/model", because Sagemaker will tar and compress all of files under "/opt/ml/model", and it will consume much time for LLM). I suggest that '/tmp/output/asset/' can be used to perform the model saving.
11. We just use the rank 0 process to upload the trained model assets to S3 by s5cmd command. It means just one of ranks will perform the thing even if multiple nodes training is used.
12. We should sync with every rank and ensure rank 0 uploading the model assets successfully (putting the torch.distributed.barrier() at the end of your taining script). Ater that, maybe there is some CUDA error when exiting the process:

          terminate called after throwing an instance of 'c10::CUDAError'
            what():  CUDA error: driver shutting down
          CUDA kernel errors might be asynchronously reported at some other API call,
          so the stacktrace below might be incorrect.
          For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
          Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:95 (most recent call first):

Please just ignore the error because the trained model assets have been uploaded to the S3.

13. When enabling RDMA protocol on EFA for P4d/P4de instance, there is very large improvement on deepspeed training speed. Just configure the following env variables in SageMaker SDK API: 'FI_PROVIDER': 'efa', 'NCCL_PROTO': 'simple', 'FI_EFA_USE_DEVICE_RDMA': '1' .  

## Deploy LLM on SageMaker

Now, we suggest that trained LLM is deployed by LMI (large model inference) container on SageMaker. LMI support 3 types accelerator: huggingface accelerate, deepspeed inference, faster transformer.

### Some useful tips:

1. For the model trained with pytorch fp16 mixed precision, the torch_dtype in the model’s config.json is also float16, but the model.dtype (such as "model = AutoModelForCausalLM.from_pretrained()") during inference is torch.float32, which is a big Pit (I spent a lot of time on this before I found out).

           Analyze:

           A. The size of saved model is about 14GB byte, and the size of model parameters are about 7B, So I inferred that the real dtype of this model's parameters is fp16. 
           When using the fp16 mixed precision for training, the final model parameter/weight is saved as fp16 or fp32 which is up to specific framework. 
           For Tensorflow, even if the model is trained with mixed precision, the dtype of saved model's parameter is also fp32 (To use mixed precision, the global policy should be set to 'mixed_float16' or 'mixed_bfloat16', so that every layer uses a 16-bit compute dtype and float32 variable dtype by default.-----Refer to the link: https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/set_global_policy); 
           but for pytorch, the saved model using the HF trainer API is fp16/bp16 after training with fp16 or bf16 mixed precision).

           B. In addition, from the torch_dtype in config.json, it is also float16. However, the model.dtype (such as "model = AutoModelForCausalLM.from_pretrained()") during inference is torch.float32, so if you directly assign model.dtype to the parameter "dytpe" in deepspeed.init_inference API, it may happen OOM issue. 
           At this time, you can set the parameter "dtype" of deepspeed.init_inference API to torch.half for fixing the OOM issue.


2. For the trained LLM such as bloomz (using deepspeed, Sagemaker model parallelism or pytorch FSDP training methods), whether using LMI on Sagemaker endpoint or direct local inference, the inference speed is always much slower than that of bloomz downloaded directly from HF (When the context text is more than 750+ tokens and the max new token is set to 128, their speed difference is 8 times). 

        Although the size of the finetuned bloomz model has become larger, the model structure, number of parameters, and dictionary size are all the same as the original HF bloomz model. The shape of the id sequence after tokenzier is also the same. 
        After setting the "use_cache" parameter to True, the inference speed becomes normal. This parameter is Only useful for inference, not for training. 
        When using the flan-t5-xxl 11B model, even if use_cache is not set to True, the difference in flan-t5-xxl's inference speed before and after finetuning is not as big as that of bloomz. Also, for this model, setting use_cache to True will also make the inference faster. 
        When using the HF model for generation, regardless of the pipeline API or the generate API, setting the parameter "use_cache" will speed up the inference speed of the model. For the explanation of "use_cache", please refer to: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958. 
        After enabling this parameter, it need not to recalculate the hiden state of all newly generated tokens when generating the next token each time, thus it will greatly save time, which is a great acceleration for the autoregressive model/causalLM.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

