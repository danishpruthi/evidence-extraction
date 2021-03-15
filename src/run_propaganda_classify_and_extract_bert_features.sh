#!/bin/sh
export PYTHONPATH="./transformers-local/src";
# joint prediction and extraction with bert feautres
mkdir -p logs;
for seed in `seq 1 5`; do 
    for k in 0.05; do
        unbuffer python main.py --data_dir ../datasets/clean_propaganda_dataset --batch_size 4 --learning_rate 2e-5 --max_seq_len 512 --epochs 10 --dataset propaganda --evaluate_every 10000 --include_bert_features --seed $seed --upper_case --gradient_accumulation_steps 8 --fraction_rationales $k | tee -a logs/logs_batch_size=4_max_seq_len=512_epochs=10_gradient_accumulation_steps=8_dataset=propaganda_include_bert_features_upper_case_seed=$seed\_fraction_rationales=$k.txt.v2;
    done;
done;
