
export CUDA_VISIBLE_DEVICES=2

python generate_mix_fonts_samples_v2.py \
--fonts_json type/songhei_font_missing.json \
--sample_dir songhei_fonts_samples/ \
--bad_fonts charset/songhei_error_font.txt \
--experiment_dir songhei_experiment \
--gpu_ids cuda \
--embedding_num 250 \
--resume 74360 \
