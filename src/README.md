# Weakly- and Semi-supervised Evidence Extraction


The `main.py` has several flags to run different types of experiments. To see what each flag does run `python main.py --help`, however, following important flags are explained below:

- `extraction_coeff` (**defaults to 1.0**) controls the coefficient of the CRF loss used for extraction
- `prediction_coeff` (**defaults to 1.0**) controls the coefficient of the cross-entropy based prediction loss
- `kld_coeff`  (**defaults to 0.0**) controls the coefficient of KL-Divergence loss which is used for attention supervision baseline 
- `include_attention_features` setting this flag allows the CRF model to use BERT's attention features 
- `include_bert_features` setting this flag allows the CRF model to use BERT's representations as features.
 
Note that you can set both `include_attention_features` and `include_bert_features`, but we didn't see much gain using both sets of features.)

Further, to condition the extraction on the predicted labels, use 

- `include_double_bert_features` this allows one to condition extraction on the predicted labels by transforming the BERT features (see the paper for details).  
- `use_oracle_labels` use this flag alongside the `include_double_bert_features` flag to condition on oracle labels


### Running Experiments

Each run script corresponds to different lines in Table 3 of the table. The name of the files describe the experiment type. For instance, `run_sentiment_classify_and_extract_attention_features.sh` corresponds to sentiment analysis task on movie reviews, and "Classify & Extract (BERT’s Attention-CRF)" setting from the paper.  