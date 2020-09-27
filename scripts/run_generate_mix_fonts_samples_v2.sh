
export CUDA_VISIBLE_DEVICES=2

python generate_mix_fonts_samples_v2.py \
--fonts_json type/kaiyuanliwei_font_missing.json \
--sample_dir kaiyuanliwei_fonts_samples/ \
--bad_fonts charset/kaiyuanliwei_error_font.txt \
--experiment_dir kaiyuanliwei_experiment \
--gpu_ids cuda \
--embedding_num 250 \
--resume 49340 \
