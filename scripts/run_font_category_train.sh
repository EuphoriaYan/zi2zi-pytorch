
export CUDA_VISIBLE_DEVICES=3

python font_category.py \
--action train \
--resume category_output/category_best.pth \
--input_path fonts_output \
--embedding_num 520 \
--type_file type/方正第二批.txt \
--epoch 50 \
--save_path category_output \
--batch_size 32 \
