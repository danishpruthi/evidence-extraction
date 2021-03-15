#!/bin/sh
# prediction only
export PYTHONPATH="./transformers-local/src"
for seed in `seq 1 5`; do
    unbuffer python main.py --data_dir ../datasets/clean_propaganda_dataset --batch_size 4 --learning_rate 2e-5 --max_seq_len 128 --epochs 10 --dataset propaganda --evaluate_every 100000 --include_bert_features --upper_case --extraction_coeff 0. --seed $seed --gradient_accumulation_steps 8 --attention_top_k 45 | tee -a logs/logs_batch_size=4_max_seq_len=128_epochs=5_dataset=propaganda_gradient_accumulation_steps=8_attention_top_k=45_include_bert_features_upper_case_extraction_coeff=0.0_seed=$seed.txt.v2;
done
