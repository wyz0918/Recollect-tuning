
#mkdir sicner_bert
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner.py  --max_seq_length 180  --max_pair_length 180 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 1 --save_steps 250 --max_mention_ori_length 8 --seed $seed \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert/sci-bert-$seed
#done;

#mkdir sicner_bert4
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner4.py  --max_seq_length 350  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 250 --max_mention_ori_length 8 --seed $seed \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert4/sci-bert-$seed
#done;

#mkdir sicner_bert7-1
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner7-1.py  --max_seq_length 190  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 15 --save_steps 150 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert7-1/sci-bert-$seed
#
#
##      --dev_file ent_pred_test_new.json \
##      --test_file ent_pred_test_new.json  \
#done;

#mkdir law_ner_bert
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner_PLM_law.py  --max_seq_length 432  --max_pair_length 20 --data_dir datasets/law  \
#      --model_name_or_path ../model/chinese-roberta-wwm-ext-large  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 2400 --max_mention_ori_length 20 --seed $seed \
#      --train_file train.json \
#      --dev_file dev.json \
#      --test_file step1_test.json  \
#      --output_dir law_ner_bert/sci-bert-$seed
##      --dev_file ent_pred_test_new.json \
##      --test_file ent_pred_test_new.json  \
#done;


#mkdir ner_bert_cancer
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner_PLM_cancer.py  --max_seq_length 432  --max_pair_length 20 --data_dir datasets/cancer  \
#      --model_name_or_path ../model/chinese-roberta-wwm-ext-large  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 2400 --max_mention_ori_length 20 --seed 42 \
#      --train_file train \
#      --dev_file dev.json \
#      --test_file dev.json  \
#      --output_dir ner_bert_cancer/bert-$seed
##      --dev_file ent_pred_test_new.json \
##      --test_file ent_pred_test_new.json  \
#done;


#mkdir sicner_bert4
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner4.py  --max_seq_length 350  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 250 --max_mention_ori_length 6 --seed $seed \
#      --train_file train_data.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert4/sci-bert-$seed
#done;
#
#mkdir sicner_bert8
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner5.py  --max_seq_length 250  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 20 --per_gpu_eval_batch_size 20 \
#      --learning_rate 5e-5 --num_train_epochs 16 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data3.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert8/sci-bert-$seed
#done;

#mkdir sicner_bert_full
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner5.py  --max_seq_length 250  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file new_train.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert_full/sci-bert-$seed
#done;
#python3 sumup.py sicner sicner_bert13/sci-bert

#mkdir sicner_bert12-test
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner6.py  --max_seq_length 250  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert12-test/sci-bert-$seed
#done;

#mkdir sicner_bert16
#for seed in 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6-5.py  --max_seq_length 165  --max_pair_length 30 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 150 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert12-5/sci-bert-$seed
#done;

#mkdir sicner_bert6-1
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6-1.py  --max_seq_length 256  --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 15 --save_steps 150 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert6-1/sci-bert-$seed
#
#done;


#mkdir sicner_bert_wo_inter
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner8_wo_interraction.py  --max_seq_length 256  --max_pair_length 32 --data_dir datasets/scierc  \
#      --fp16 --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert_wo_inter/sci-bert-$seed
#done;

#mkdir sicner_bert13-ori-80-6
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8.py  --fp16 --max_seq_length 200  --max_pair_length 40 --data_dir datasets/scierc  \
#      --do_test --do_train --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 1800 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-80-6/sci-bert-$seed
#done;

#mkdir sicner_bert13-ori-80-6-base
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner8.py  --fp16 --max_seq_length 200  --evaluate_during_training --max_pair_length 40 --data_dir datasets/scierc  \
#      --do_test --do_train --do_eval --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 1800 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-80-6-base/sci-bert-$seed
#done;

#mkdir scire_bert_ratio_10-base
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10.py  \
#      --do_test --do_train --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-80-6-base/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10-base/scire-bert-$seed
#done;

#mkdir sicner_bert13-ori-80-8
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8_sci.py  --fp16 --max_seq_length 200  --max_pair_length 40 --data_dir datasets/scierc  \
#      --do_test --do_train --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 2400 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-80-8/sci-bert-$seed
#done;


#mkdir sicner_bert13-ori-80-9
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner8_sci.py  --fp16 --max_seq_length 200  --max_pair_length 40 --data_dir datasets/scierc  \
#      --do_test --do_train --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 8 --save_steps 1800 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-80-9/sci-bert-$seed
#done;
#

##




#
#
#mkdir sicner_bert13-ori-0%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner0.py  --max_seq_length 256 --max_pair_length 40 --data_dir ../datasets/scierc  \
#      --do_test --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-0%/sci-bert-$seed
#done;


#mkdir sicner_bert13-ori-20%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner2.py  --max_seq_length 256 --max_pair_length 40 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-20%/sci-bert-$seed
#done;

#mkdir sicner_bert13-ori-40%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner4.py  --max_seq_length 256  --max_pair_length 40 --data_dir ../datasets/scierc  \
#      --do_test --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-40%/sci-bert-$seed
#done;


#mkdir sicner_bert13-ori-60%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6.py  --max_seq_length 256  --max_pair_length 40 --data_dir ../datasets/scierc  \
#      --do_test --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-60%/sci-bert-$seed
#done;
#
#mkdir sicner_bert13-ori-100%
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner10.py  --max_seq_length 256  --max_pair_length 25 --data_dir ../datasets/scierc  \
#      --do_train --do_test --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-100%/sci-bert-$seed
#done;

#mkdir ace05ner_pipeline2
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8_ace05_ner.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05_2 \
#      --fp16 --do_train --do_test --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 1200 --max_mention_ori_length 12 --seed $seed \
#      --train_file train.json \
#      --dev_file dev.json \
#      --test_file test.json  \
#      --output_dir ace05ner_pipeline2/bert-$seed
#done;


#mkdir ace05ner_pipeline
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner8.py  --max_seq_length 256  --max_pair_length 32 --data_dir ./  \
#      --do_test --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_pipeline/bert-$seed
#done;

#mkdir ace05ner_random
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8_random.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --do_train --do_test --do_eval --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_random/bert-$seed
#done;
#
#mkdir ace05ner_bert_wo_inter_random
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  ./ablationStudy/NER/Random/run_ner8_wo_interraction_random.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05   \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert_wo_inter_random/sci-bert-$seed
#done;
#
#mkdir ace05ner_random_0
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner0_random.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --do_train --do_test --do_eval --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_random_0/bert-$seed
#done;

#mkdir sciner_random
#for seed in 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner8_random.py  --fp16 --max_seq_length 256  --max_pair_length 32  \
#      --do_train --do_test --do_eval --model_name_or_path ../model/scibert_scivocab_uncased  --data_dir datasets/scierc --model_type bertspanmarker \
#      --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sciner_random/bert-$seed
#done;

#mkdir sciner_bert_wo_inter_random
#for seed in 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8_wo_interraction_random.py  --fp16 --max_seq_length 256  --max_pair_length 32   \
#      --model_name_or_path ../model/scibert_scivocab_uncased  --data_dir datasets/scierc --model_type bertspanmarker \
#      --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sciner_bert_wo_inter_random/sci-bert-$seed
#done;

#mkdir sciner_random_0
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner0_random.py  --fp16 --max_seq_length 256  --max_pair_length 32   \
#      --do_train --do_test --do_eval --model_name_or_path bert_models/scibert_scivocab_uncased  --data_dir datasets/scierc   --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir sciner_random_0/bert-$seed
#done;
#



#mkdir ace05ner_random_XX
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner8_random.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ./  \
#      --do_train --do_test --do_eval --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_random/bert-$seed
#done;
#
#mkdir ace05ner_bert_wo_inter_random_XX
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8_wo_interraction_random.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert_wo_inter_random/sci-bert-$seed
#done;
#
#mkdir ace05ner_random_0_XX
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner0_random.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ./  \
#      --do_train --do_test --do_eval --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_random/bert-$seed
#done;




#mkdir ace05ner_bert_wo_inter
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner8_wo_interraction.py  --fp16 --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert_wo_inter/sci-bert-$seed
#done;

#mkdir ace05ner_bert13-ori-80%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner8.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13-ori-80%/sci-bert-$seed
#done;
#
#mkdir ace05ner_bert13--60%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner6.py  --max_seq_length 256  --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13--60%/sci-bert-$seed
#done;
#
#mkdir ace05ner_bert13--40%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner4.py  --max_seq_length 256  --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13--40%/sci-bert-$seed
#done;

#mkdir ace05ner_bert13-ori-20%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner2.py  --max_seq_length 256 --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13-ori-20%/sci-bert-$seed
#done;


#mkdir ace05ner_bert13_75%_l2r
#for seed in 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner6_l2r.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --seed $seed --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 250 --max_mention_ori_length 8 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13_75%_l2r/sci-bert-$seed
#done;

#mkdir ace05ner_bert13_75%_r2l
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6_r2l.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --seed $seed  --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 250 --max_mention_ori_length 8 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13_75%_r2l/sci-bert-$seed
#done;
#
#mkdir ace05ner_bert13_75%_c2b
#for seed in 46; do
#CUDA_VISIBLE_DEVICES=1 python3  run_ner6_c2b.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --seed $seed  --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 250 --max_mention_ori_length 8 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13_75%_c2b/sci-bert-$seed
#done;

#mkdir ace05ner_bert13_75%_b2c
#for seed in 46; do
#CUDA_VISIBLE_DEVICES=2 python3  run_ner6_b2c.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --seed $seed  --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 250 --max_mention_ori_length 8 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13_75%_b2c/sci-bert-$seed
#done;

#mkdir ace05ner_bert13_75%_lowdata
#for seed in 20 40 60 80 100; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --data_volume $seed --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 250 --max_mention_ori_length 8 --seed 42 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13_75%_lowdata/sci-bert-$seed
#done;

#mkdir ace05ner_bert13_75%_lowdata
#for seed in 100; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --data_volume $seed --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 250 --max_mention_ori_length 8 --seed 42 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13_75%_lowdata/sci-bert-$seed
#done;



#mkdir ace05ner_bert13-ori-wo-75%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6_without.py  --max_seq_length 256  --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 15 --save_steps 500 --max_mention_ori_length 8 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13-ori-wo-75%/sci-bert-$seed
#done;

##
#mkdir ace05ner_albert_80_2
#  for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner8.py  --fp16 --max_seq_length 256  --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/albert-xxlarge-v2  --model_type albertspanmarker  \
#      --max_pair_length 40 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 1500 --max_mention_ori_length 10  --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_albert_80_2/ace05-albert-$seed
#done;


#mkdir ace04ner_bert_75
#for data_spilt in 1; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6.py  --max_seq_length 384  --max_pair_length 40 --data_dir ../datasets/ace04  \
#      --do_train --do_test --do_eval  --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 6 --seed 42 \
#      --train_file train/$data_spilt.json \
#      --dev_file dev/$data_spilt.json \
#      --test_file test/$data_spilt.json  \
#      --output_dir ace04ner_bert_75/ace04-bert-$data_spilt
#done;

#mkdir ace04ner_bert_75_XX
#for data_spilt in 0 1 2 3 4; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner6.py  --max_seq_length 300  --max_pair_length 40 --data_dir ../datasets/ace04  \
#      --model_name_or_path ../model/albert-xxlarge-v2  --model_type albertspanmarker \
#      --per_gpu_train_batch_size 6 --per_gpu_eval_batch_size 6 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 1000 --max_mention_ori_length 6 --seed 42 \
#      --train_file train/$data_spilt.json \
#      --dev_file dev/$data_spilt.json \
#      --test_file test/$data_spilt.json  \
#      --output_dir ace04ner_bert_75_XX/ace04-bert-$data_spilt
#done;


#mkdir ace05ner_bert13-ori-speedup2-75%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6_wo_speedup.py  --max_seq_length 256  --max_pair_length 32 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 2700 --max_mention_ori_length 8 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13-ori-speedup2-75%/sci-bert-$seed
#done;

#mkdir ace05ner_PLMarker_lowdata
#for seed in 20 40 60 80 100; do
#CUDA_VISIBLE_DEVICES=3 python3  ace05ner_test.py  --max_seq_length 512  --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --data_volume $seed --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 2500 --max_mention_ori_length 8 --seed 42 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_PLMarker_lowdata/sci-bert-$seed
#done;

#mkdir ace05ner_bert_Gold_Only
#for seed in 20 40 60 80 100; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner0.py  --max_seq_length 256 --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --data_volume $seed --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed 42 \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert_Gold_Only/sci-bert-$seed
#done;



#mkdir ace05ner_bert13-ori-0%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner0.py  --max_seq_length 256 --max_pair_length 40 --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert13-ori-0%/sci-bert-$seed
#done;
#
##
#
mkdir ace05ner_bert13-ori-100%
for seed in 42 43 44 45 46; do
CUDA_VISIBLE_DEVICES=0 python3  run_ner10.py  --max_seq_length 256  --max_pair_length 25 --data_dir ../datasets/ace05  \
      --do_train --do_test --do_eval  --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 10 --seed $seed \
      --train_file train_data.json \
      --dev_file dev_data.json \
      --test_file test_data.json  \
      --output_dir ace05ner_bert13-ori-100%/sci-bert-$seed
done;


#mkdir sicner_bert13-ori-2
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner7.py  --max_seq_length 200  --max_pair_length 30 --data_dir datasets/scierc  \
#      --model_name_or_path bert_models/scibert_scivocab_uncased  --data_dir datasets/scierc  --model_type bertspanmarker \
#      --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 5e-5 --num_train_epochs 10 --save_steps 300 --max_mention_ori_length 10 --seed $seed \
#      --train_file train_data.json \
#      --dev_file test_data.json \
#      --test_file test_data.json  \
#      --output_dir sicner_bert13-ori-2/sci-bert-$seed
#done;

#ACE05
#mkdir ace05ner_bert5
#  for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3 python3  run_ner7.py  --max_seq_length 232  --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/bert-base-uncased  --model_type bertspanmarker \
#      --max_pair_length 35 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 500 --max_mention_ori_length 8  --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_bert5/ace05-bert-$seed
#done;

#mkdir ace05ner_albert3
#  for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0 python3  run_ner6.py  --max_seq_length 256  --data_dir ../datasets/ace05  \
#      --model_name_or_path ../model/albert-xxlarge-v2  --model_type albertspanmarker  \
#      --max_pair_length 40 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
#      --learning_rate 2e-5 --num_train_epochs 10 --save_steps 2000 --max_mention_ori_length 8  --seed $seed \
#      --train_file train_data.json \
#      --dev_file dev_data.json \
#      --test_file test_data.json  \
#      --output_dir ace05ner_albert3/ace05-albert-$seed
#done;

