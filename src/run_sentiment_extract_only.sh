#!/bin/sh
# extraction only
export PYTHONPATH="./transformers-local/src";
for seed in `seq 1 5`; do
    unbuffer python main.py --data_dir ../datasets/movie_reviews_only_rats --batch_size 4 --learning_rate 2e-5 --max_seq_len 512 --epochs 10 --dataset movie_reviews --evaluate_every 100000 --include_bert_features --upper_case --gradient_accumulation_steps 8 --prediction_coeff 0. --dump_rationales --seed $seed | tee -a logs/logs_batch_size=4_max_seq_len=512_epochs=10_gradient_accumulation_steps=8_dataset=movie_reviews_include_bert_features_upper_case_prediction_coeff=0.0_dump_rationales_seed=$seed.txt;
done
