

src_fonts_dir='charset/ZhongHuaSong'
dst_imgs=shufa_pic/square_img

python font2img.py \
--src_fonts_dir ${src_fonts_dir} \
--dst_imgs ${dst_imgs} \
--sample_count 200000 \
--sample_dir ./shufa_samples \
--mode fonts2imgs
