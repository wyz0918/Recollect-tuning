


# mkdir ace04re_bert_100
#  for data_spilt in 0 1 2 3 4; do
#      CUDA_VISIBLE_DEVICES=1 python3  run_re10.py  \
#      --do_test --gold --max_seq_length 180    \
#      --model_name_or_path ../model/bert-base-uncased  \
#      --model_type bertspanmarker --max_pair_length 35 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 500 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_100/ace04re-bert-$data_spilt
#  done;
#
#
# mkdir ace04re_bert_2
#  for data_spilt in 0 1 2 3 4; do
#      CUDA_VISIBLE_DEVICES=0 python3  run_re2.py  \
#      --do_train --do_test --max_seq_length 180    \
#      --model_name_or_path ../model/bert-base-uncased  \
#      --model_type bertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 500 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_2/ace04re-bert-$data_spilt
#  done;
#
#
# mkdir ace04re_bert_100XX
#  for data_spilt in 0; do
#      CUDA_VISIBLE_DEVICES=0 python3  run_re10_2.py  \
#      --do_test --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --model_type albertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 5000 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75_XX/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_100XX/ace04re-bert-$data_spilt
#  done;

# mkdir ace04re_bert_100XX_2
#  for data_spilt in 1; do
#      CUDA_VISIBLE_DEVICES=2 python3  run_re10.py  \
#      --do_test --do_train --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --model_type albertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 4500 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75_XX/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_100XX_2/ace04re-bert-$data_spilt
#  done;

# mkdir ace04re_bert_100XX_2
#  for data_spilt in 3; do
#      CUDA_VISIBLE_DEVICES=2 python3  run_re10.py  \
#      --do_test --do_train --fp16 --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --model_type albertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 6  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 2500 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75_XX/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_100XX_2/ace04re-bert-$data_spilt
#  done;

# mkdir ace04re_bert_100XX_3
#  for data_spilt in 3; do
#      CUDA_VISIBLE_DEVICES=2 python3  run_re10.py  \
#      --do_test --do_train --fp16 --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --model_type albertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 8000 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75_XX/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_100XX_3/ace04re-bert-$data_spilt
#  done;

# mkdir ace04re_bert_100XX_6
#  for data_spilt in 3; do
#      CUDA_VISIBLE_DEVICES=0 python3  run_re10.py  \
#      --do_test --do_train --fp16 --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --model_type albertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 10  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace04 --save_steps 800 \
#      --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  \
#      --test_file ace04ner_bert_75_XX/ace04-bert-$data_spilt/ent_pred_test.json   \
#      --output_dir ace04re_bert_100XX_6/ace04re-bert-$data_spilt
#  done;


# mkdir ace05re_bert_40XX2
#for seed in 44 45 46; do
#      CUDA_VISIBLE_DEVICES=0 python3  run_re10.py  \
#      --fp16 --do_test --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --learning_rate 1e-5 --model_type albertspanmarker --max_pair_length 35 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16  --num_train_epochs 12 \
#      --data_dir ../datasets/ace05 --save_steps 1000 \
#      --dev_file ace05ner_albert_40/ace05-albert-$seed/ent_pred_test.json   \
#      --test_file ace05ner_albert_40/ace05-albert-$seed/ent_pred_test.json   \
#      --output_dir ace05re_bert_40XX2/ace05re-bert-$seed
#  done;


# mkdir ace05re_bert_2XX
#for seed in 42 43 44 45 46; do
#      CUDA_VISIBLE_DEVICES=1 python3  run_re2.py  \
#      --do_train --do_test --max_seq_length 180    \
#      --model_name_or_path ../model/albert-xxlarge-v2 \
#      --model_type albertspanmarker --max_pair_length 40 --per_gpu_train_batch_size 8  \
#      --per_gpu_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 \
#      --data_dir ../datasets/ace05 --save_steps 1000 \
#      --test_file ace05ner_albert3/ace05-albert-$seed/ent_pred_test.json   \
#      --output_dir ace05re_bert_2XX/ace05re-bert-$seed
#  done;


#mkdir ace05re_bert_Gold_Only
#for seed in 20 40 60 80 100; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re0.py  \
#       --data_volume $seed  --gold --do_test --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed 42 \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert_Gold_Only/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_Gold_Only/scire-bert-$seed
#done;



#mkdir ace05re_bert_20_2%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re2.py  \
#       --do_train --do_test --max_seq_length 256  --max_pair_length 40  --save_steps 413  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-75%-4/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_20_2%/scire-bert-$seed
#done;

#mkdir ace05re_PLM
# for seed in 42 43 44 45 46; do
# CUDA_VISIBLE_DEVICES=2 python3 run_re_test.py  --model_type bertsub  \
#     --model_name_or_path ../model/bert-base-uncased  --do_lower_case  \
#     --data_dir ../datasets/ace05  --use_ner_results \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 32  --save_steps 5000  \
#     --do_test  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
#     --seed $seed    \
#     --test_file ace05ner_bert13-ori-PL-Marker-75%/sci-bert-$seed/ent_pred_test.json   \
#     --output_dir ace05re_PLM/ace05re-bert-$seed  --overwrite_output_dir
# done;





#mkdir ace05re_bert_20_3_wo%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re2_wo.py  \
#       --gold --do_test --max_seq_length 180  --max_pair_length 40  --save_steps 413  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-wo-75%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_20_3_wo%/scire-bert-$seed
#done;
#
#mkdir ace05re_bert_20_3%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re2.py  \
#       --gold --do_test --max_seq_length 180  --max_pair_length 40  --save_steps 413  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-75%-4/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_20_3%/scire-bert-$seed
#done;


#mkdir ace05re_bert_100_lowdata2
#for seed in 20 40 60 80 100; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10.py  \
#       --data_volume $seed  --gold --do_test --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed 42 \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13_75%_lowdata/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_100_lowdata2/scire-bert-$seed
#done;

#mkdir ace05re_bert_20_lowdata2
#for seed in 20 40 60 80 100; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re2.py  \
#       --data_volume $seed --gold --do_test --max_seq_length 180  --max_pair_length 40  --save_steps 200  --seed 42 \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13_75%_lowdata/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_20_lowdata2/scire-bert-$seed
#done;


#mkdir ace05re_bert_20_4%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re2.py  \
#       --do_train --do_test --max_seq_length 180  --max_pair_length 40  --save_steps 413  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 20 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-75%-4/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_20_4%/scire-bert-$seed
#done;

#mkdir ace05re_bert_40%
#for seed in 45 46; do
#CUDA_VISIBLE_DEVICES=1  python3 run_re4.py  \
#       --gold --do_test --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-40%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_40%/scire-bert-$seed
#done;

#mkdir ace05re_bert_60%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re6.py  \
#       --gold --do_test --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-60%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_60%/scire-bert-$seed
#done;
#
#mkdir ace05re_bert_80%
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re8.py  \
#       --gold --do_test --max_seq_length 256  --max_pair_length 32  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_80%/scire-bert-$seed
##done;
##
#mkdir ace05re_bert_100%
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re10.py  \
#       --do_test --max_seq_length 256  --max_pair_length 25  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-100%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_100%/scire-bert-$seed
#done;


#mkdir scire_bert_ratio_10
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re10.py  \
#      --do_test --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-60%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10/scire-bert-$seed
#done;
#
#mkdir scire_bert_ratio_10-base
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2  python3 run_re10.py  \
#      --do_test --do_train --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-80-6-base/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10-base/scire-bert-$seed
#done;
#mkdir scire_bert_ratio_10_wo_inter
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10_wo_inter.py  \
#      --do_test --gold --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert_wo_inter/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_wo_inter/scire-bert-$seed
#done;
# mkdir ace05re_bert_ratio_20
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re2.py  \
#       --gold  --do_test --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13-ori-20%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_20/scire-bert-$seed
#done;


#mkdir ace05re_bert_ratio_40_2
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re4.py  \
#       --gold  --do_test --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 15 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13--80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_40_2/scire-bert-$seed
#done;
##
##
##
##
#mkdir ace05re_bert_ratio_60
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2  python3 run_re6.py  \
#       --gold   --do_test --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 15 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13--80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_60/scire-bert-$seed
#done;

#mkdir ace05re_bert_ratio_100_17
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1  python3 run_re10.py  \
#       --gold --do_test  --fp16 --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 300  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 45   \
#       --per_gpu_eval_batch_size 45 --learning_rate 5e-5 --num_train_epochs 9 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13--80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_17/scire-bert-$seed
#done;

#mkdir ace05re_bert_ratio_100_6
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re10.py  \
#       --do_train --do_test --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 330  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 40   \
#       --per_gpu_eval_batch_size 40 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13--80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_6/scire-bert-$seed
#done;
#mkdir ace05re_bert_ratio_100_6
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=2  python3 run_re10.py  \
#       --do_train   --do_test --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 1200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 8 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13--80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_6/scire-bert-$seed
#done;
#mkdir ace05re_bert_ratio_80
#for seed in 42 do
#CUDA_VISIBLE_DEVICES=0  python3 run_re8.py  \
#       --gold   --do_test --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13--80%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_80/scire-bert-$seed
#
#

#w/o masked copy

mkdir ace05re_bert_0
for seed in 42; do
CUDA_VISIBLE_DEVICES=0  python3 run_re0.py  \
       --do_test --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed $seed \
       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
       --dev_file dev_data.json \
       --test_file ace05ner_bert_wo_inter/sci-bert-42/ent_pred_test.json\
       --output_dir ace05re_bert_0/scire-bert-$seed
done;
#
#mkdir scire_bert_ratio_0
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re0.py  \
#     --do_test --max_seq_length 180  --max_pair_length 20  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-0%/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_0/scire-bert-$seed
#done;
##
#
#mkdir scire_bert_ratio_10_1
#for seed in 42 43 44; do
#CUDA_VISIBLE_DEVICES=2  python3 run_re10.py  \
#      --do_test --do_train --max_seq_length 180  --max_pair_length 35  --save_steps 2400  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-80-6/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_1/scire-bert-$seed
#done;



#mkdir scire_bert_ratio_10_2
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10.py  \
#      --do_test --do_train --max_seq_length 180  --max_pair_length 35  --save_steps 1800  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-80-6/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_2/scire-bert-$seed
#done;

#mkdir scire_bert_ratio_10_3
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=1  python3 run_re10.py  \
#      --do_test --do_train --max_seq_length 180  --max_pair_length 35  --save_steps 1600  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert13-ori-80-6/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_3/scire-bert-$seed
#
#


#mkdir ace05re_bert_ratio_10_random
#for seed in 44; do
#CUDA_VISIBLE_DEVICES=2  python3 run_re10_random.py  \
#       --do_test --gold --fp16 --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 15 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_random/bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_10_random/scire-bert-$seed
#done;

# mkdir scire_bert_ratio_10_random
#for seed in 44; do
#CUDA_VISIBLE_DEVICES=0 python3 run_re10_random.py  \
#      --do_test --gold --fp16 --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sciner_random/bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_random/scire-bert-$seed
#done;

#
# aaaaa
# mkdir ace05re_bert_0_random
#for seed in 44; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re0_random.py  \
#       --do_test --gold --fp16 --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 24   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_random_0/bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_0_random/scire-bert-$seed
#done;
#
#mkdir ace05re_bert_10_wo_inter_random1
#for seed in 44; do
#CUDA_VISIBLE_DEVICES=0 python3 ./ablationStudy/RE/Random/run_re10_wo_inter_random.py  \
#       --do_test --gold --fp16 --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert_wo_inter_random/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_10_wo_inter_random1/scire-bert-$seed
#done;


#
#
#mkdir scire_bert_ratio_0_random
#for seed in 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re0_random.py  \
#     --do_test --gold --max_seq_length 180  --max_pair_length 20  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sciner_random_0/bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_0_random/scire-bert-$seed
#done;
#
#mkdir scire_bert_ratio_10_wo_inter_random
#for seed in 44; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re10_wo_inter_random.py  \
#      --do_test --gold --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sciner_bert_wo_inter_random/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_wo_inter_random/scire-bert-$seed
#done;


###w/o interaction
##
#mkdir scire_bert_ratio_10_wo_inter
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re10_wo_inter.py  \
#      --do_test --max_seq_length 180  --max_pair_length 35  --save_steps 200  --seed $seed \
#     --data_dir ../datasets/scierc --model_name_or_path ../model/scibert_scivocab_uncased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file sicner_bert_wo_inter/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir scire_bert_ratio_10_wo_inter/scire-bert-$seed
#done;
#
#
#
#mkdir ace05re_bert_10_wo_inter
#for seed in 42; do
#CUDA_VISIBLE_DEVICES=0  python3 run_re10_wo_inter.py  \
#       --do_test --fp16 --max_seq_length 256  --max_pair_length 40  --save_steps 400  --seed $seed \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert_wo_inter/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_10_wo_inter/scire-bert-$seed
#done;


#mkdir ace05re_bert_ratio_100_l2r
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10_l2r.py  \
#       --do_test  --gold --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13_75%_l2r/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_l2r/scire-bert-$seed
#done;
##
##
##
#mkdir ace05re_bert_ratio_100_r2l
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10_r2l.py  \
#       --do_test --gold --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13_75%_r2l/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_r2l/scire-bert-$seed
#done;
#
#
# mkdir ace05re_bert_ratio_100_c2b
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10_c2b.py  \
#       --do_test --gold --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13_75%_c2b/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_c2b/scire-bert-$seed
#done;




#mkdir ace05re_bert_ratio_100_b2c
#for seed in 42 43 44 45 46; do
#CUDA_VISIBLE_DEVICES=3  python3 run_re10.py  \
#       --do_test --do_train --fp16 --seed $seed --max_seq_length 180  --max_pair_length 35  --save_steps 200  \
#       --data_dir ../datasets/ace05 --model_name_or_path ../model/bert-base-cased  \
#       --model_type bertspanmarker --per_gpu_train_batch_size 32   \
#       --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10 \
#       --dev_file dev_data.json \
#       --test_file ace05ner_bert13_75%_b2c/sci-bert-$seed/ent_pred_test.json  \
#       --output_dir ace05re_bert_ratio_100_b2c/scire-bert-$seed
#done;
