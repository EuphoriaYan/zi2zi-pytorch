
export CUDA_VISIBLE_DEVICES=3

python massive_train.py \
--experiment_dir songhei_experiment \
--gpu_ids cuda:0 \
--embedding_num 250 \
--epoch 20 \
--batch_size 128 \
--sample_steps 10000 \
--checkpoint_steps 10000 \
--input_nc 1 \
