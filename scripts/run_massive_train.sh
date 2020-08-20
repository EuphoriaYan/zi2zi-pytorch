
export CUDA_VISIBLE_DEVICES=3

python massive_train.py \
--experiment_dir experiment \
--massive_id 01 \
--gpu_ids cuda:0 \
--embedding_num 205 \
--epoch 20 \
--batch_size 64 \
--sample_steps 50000 \
--checkpoint_steps 50000 \
--input_nc 1 \
