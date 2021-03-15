#!/bin/sh
# ----------------------------------  basic tests ---------------------------------- #
export PYTHONPATH="./transformers-local/src";
# only extraction using bert features
for seed in `seq 1 5`; do
    for k in 0.05; do
        unbuffer python main.py --data_dir ../datasets/clean_propaganda_dataset --batch_size 4 --learning_rate 2e-5 --max_seq_len 128 --epochs 10 --dataset propaganda --evaluate_every 100000 --include_bert_features --upper_case --prediction_coeff 0. --seed $seed --fraction_rationales $k --gradient_accumulation_steps 8 | tee -a logs/logs_batch_size=4_max_seq_len=128_epochs=10_dataset=propaganda_include_bert_features_upper_case_prediction_coeff=0.0_seed=$seed\_fraction_rationales=$k.txt.v2;
    done;
done
