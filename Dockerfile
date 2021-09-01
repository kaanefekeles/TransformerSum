FROM nvidia/cuda:11.4.1-base-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip

COPY . /TransformerSum/

WORKDIR /TransformerSum

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install -r requirements.txt

WORKDIR /TransformerSum/src/

CMD  python3 -W ignore main.py --mode abstractive --model_name_or_path dbmdz/bert-base-turkish-uncased --decoder_model_name_or_path dbmdz/bert-base-turkish-uncased --cache_file_path data --max_epochs 4  --do_train --do_test  --batch_size 4  --weights_save_path model_weights --no_wandb_logger_log_model --accumulate_grad_batches 5 --use_scheduler linear --warmup_steps 8000 --gradient_clip_val 1.0 --custom_checkpoint_every_n 300 --data_example_column text --data_summarized_column summary --gpus -1
