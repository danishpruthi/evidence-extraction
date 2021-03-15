#!/bin/sh
export PYTHONPATH="./transformers-local/src";
# joint prediction and extraction using attention features
for seed in `seq 1 5`; do 
    unbuffer python main.py --data_dir ../datasets/movie_reviews_with_some_rats --batch_size 4 --learning_rate 2e-5 --max_seq_len 512 --epochs 10 --dataset movie_reviews --evaluate_every 10000 --include_attention_features --seed $seed --upper_case --gradient_accumulation_steps 8 | tee -a logs/logs_batch_size=4_max_seq_len=512_epochs=10_gradient_accumulation_steps=8_dataset=movie_reviews_include_attention_features_upper_case_seed=$seed.txt.v2;
done;
