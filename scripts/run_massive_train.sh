
export CUDA_VISIBLE_DEVICES=2

python massive_train.py \
--experiment_dir massive_experiment3 \
--start_from 601 \
--gpu_ids cuda:0 \
--embedding_num 205 \
--epoch 20 \
--batch_size 64 \
--sample_steps 10000 \
--checkpoint_steps 10000 \
--input_nc 1 \
