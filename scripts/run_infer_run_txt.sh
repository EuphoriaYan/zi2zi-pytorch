

python infer.py \
--experiment_dir songhei_experiment \
--embedding_num 250 \
--gpu_ids cuda:0 \
--resume 74360 \
--batch_size 32 \
--from_txt \
--src_txt 我爱北京天安门 \
--label 123 \
--type_file type/宋黑类字符集.txt \
