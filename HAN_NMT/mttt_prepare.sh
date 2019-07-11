#!/bin/bash

main_dir=/data/tomasz/baselines/HAN_NMT/datasets/
language=en-zh
src=zh
tgt=en

data_dir=$main_dir/$language/tok

python full_source/preprocess.py -train_src $data_dir/ted_train_$language.tok.clean.$src -train_tgt $data_dir/ted_train_$language.tok.clean.$tgt -train_doc $data_dir/ted_train_$language.tok_doc -valid_src $data_dir/ted_dev_$language.tok.$src -valid_tgt $data_dir/ted_dev_$language.tok.$tgt -valid_doc $data_dir/ted_dev_$language.tok_doc -save_data $main_dir/$language/model_in/mttt_$src\-$tgt -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80
