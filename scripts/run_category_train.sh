
export CUDA_VISIBLE_DEVICES=3

python category.py \
--action train \
--resume fz2_experiment/checkpoint/187920_net_D.pth \
--input_path fonts_output \
--embedding_num 520 \
--type_file type/方正第二批.txt \
--epoch 30 \
--save_path category_output \
