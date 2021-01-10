
export CUDA_VISIBLE_DEVICES=3

python shufa_category.py \
--action train \
--resume category_output/category_best.pth \
--input_path shufa_pic/square_img \
--embedding_num 30 \
--epoch 10 \
--save_path shufa_category_output \
--batch_size 32 \
