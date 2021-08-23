FROM nvidia/cuda:11.4.1-base-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip && apt-get install -y git

RUN git clone https://github.com/kaanefekeles/TransformerSum

WORKDIR /TransformerSum

RUN pip install -r requirements.txt

RUN wandb offline

WORKDIR /TransformerSum/src/

RUN python3 main.py --mode abstractive --model_name_or_path dbmdz/bert-base-turkish-uncased --decoder_model_name_or_path dbmdz/bert-base-turkish-uncased --cache_file_path data --max_epochs 4  --do_train --do_test  --batch_size 4  --weights_save_path model_weights --no_wandb_logger_log_model --accumulate_grad_batches 5 --use_scheduler linear --warmup_steps 8000 --gradient_clip_val 1.0 --custom_checkpoint_every_n 300 --data_example_column text --data_summarized_column summary --gpus 0
