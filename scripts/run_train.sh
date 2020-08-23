
export CUDA_VISIBLE_DEVICES=2

python train.py \
--experiment_dir ZhongHuaSong2Shufa_experiment \
--gpu_ids cuda:0 \
--epoch 50 \
--batch_size 64 \
--sample_steps 10000 \
--checkpoint_steps 10000 \
--input_nc 1 \
