#!/bin/sh
# only prediction
export PYTHONPATH="./transformers-local/src"
for seed in `seq 1 1`; do 
    unbuffer python3 main.py --data_dir ../datasets/movie_reviews_with_some_rats --batch_size 4 --learning_rate 2e-5 --max_seq_len 512 --epochs 10 --dataset movie_reviews --evaluate_every 10000 --include_bert_features --seed $seed --upper_case --extraction_coeff 0.0 --gradient_accumulation_steps 8 | tee -a logs/logs_batch_size=4_max_seq_len=512_epochs=10_gradient_accumulation_steps=8_dataset=movie_reviews_include_bert_features_upper_case_extraction_coeff=0.0_seed=$seed.txt;
done;
