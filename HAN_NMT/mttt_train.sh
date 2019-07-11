#!/bin/bash

main_dir_data=/data/tomasz/baselines/HAN_NMT/datasets/
main_dir_model=/data/tomasz/baselines/HAN_NMT/models/

language=en-zh
src=zh
tgt=en
gpu_device=4

# Training the sentence-level NMT baseline
CUDA_VISIBLE_DEVICES=$gpu_device python full_source/train.py -data $main_dir_data/$language/model_in/mttt_$src\-$tgt -save_model $main_dir_model/$language/sentence_$src\-$tgt -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 20 -report_every 500 -epochs 20 -gpuid 0 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part sentences

sentence_model=$(ls $main_dir_model/$language/sentence_$src\-$tgt_*_e20.pt)

# Training HAN-encoder using the sentence-level NMT model 
CUDA_VISIBLE_DEVICES=$gpu_device python full_source/train.py -data $main_dir_data/$language/model_in/mttt_$src\-$tgt -save_model $main_dir_model/$language/HAN_enc_model_$src\-$tgt -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_enc -context_size 3 -train_from $sentence_model

# Training HAN-decoder using the sentence-level NMT model
CUDA_VISIBLE_DEVICES=$gpu_device python full_source/train.py -data $main_dir_data/$language/model_in/mttt_$src\-$tgt  -save_model $main_dir_model/$language/HAN_dec_model -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_dec -context_size 3 -train_from $sentence_model

han_enc_model=$(ls $main_dir_model/$language/HAN_enc_model_$src\-$tgt_*_e20.pt)

# Training HAN-joint using the HAN-encoder model
CUDA_VISIBLE_DEVICES=$gpu_device python full_source/train.py -data $main_dir_data/$language/model_in/mttt_$src\-$tgt -save_model $main_dir_model/$language/HAN_joint_model -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_join -context_size 3 -train_from $han_enc_model

