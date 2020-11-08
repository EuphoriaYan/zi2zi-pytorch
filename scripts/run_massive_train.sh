
export CUDA_VISIBLE_DEVICES=3

python massive_train.py \
--experiment_dir fz2_experiment \
--gpu_ids cuda \
--embedding_num 520 \
--epoch 20 \
--batch_size 128 \
--sample_steps 10000 \
--checkpoint_steps 10000 \
--input_nc 1 \
