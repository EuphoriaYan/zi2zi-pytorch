
export CUDA_VISIBLE_DEVICES=3

python generate_mix_fonts_samples.py \
--create_num 0 \
--fonts_json /disks/sdb/projs/AncientBooks/data/chars/font_missing.json \
--sample_dir fonts_samples/ \
--experiment_dir massive_experiment0 \
--gpu_ids cuda \
--embedding_num 205 \
--resume 300000 \
--start_from 23 \
