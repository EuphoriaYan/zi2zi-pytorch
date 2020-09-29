
export CUDA_VISIBLE_DEVICES=2

python generate_mix_fonts_samples_v2.py \
--fonts_json type/caokai_font_missing.json \
--sample_dir caokai_fonts_samples/ \
--bad_fonts charset/caokai_error_font.txt \
--experiment_dir caokai_experiment \
--gpu_ids cuda \
--embedding_num 250 \
--resume 46800 \
