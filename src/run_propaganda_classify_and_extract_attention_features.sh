#!/bin/sh
# ----------------------------------  basic tests ---------------------------------- #
# joint prediction and extraction with attention feautres
export PYTHONPATH="./transformers-local/src";
mkdir -p logs;
for seed in `seq 1 5`; do 
    for k in 0.05; do
        unbuffer python main.py --data_dir ../datasets/clean_propaganda_dataset --batch_size 4 --learning_rate 2e-5 --max_seq_len 512 --epochs 10 --dataset propaganda --evaluate_every 10000 --include_attention_features --seed $seed --upper_case --gradient_accumulation_steps 8 --fraction_rationales $k --attention_top_k 45 | tee -a logs/logs_batch_size=4_max_seq_len=512_epochs=10_gradient_accumulation_steps=8_dataset=propaganda_attention_top_k=45_include_attention_features_upper_case_seed=$seed\_fraction_rationales=$k.txt.v2.new;
    done;
done;
