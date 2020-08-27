
export CUDA_VISIBLE_DEVICES=1

python massive_train.py \
--experiment_dir massive_experiment2 \
--start_from 401 \
--gpu_ids cuda:0 \
--embedding_num 205 \
--epoch 20 \
--batch_size 64 \
--sample_steps 10000 \
--checkpoint_steps 10000 \
--input_nc 1 \
