import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

if __name__ == '__main__':
    jpg_path = '../shufa_pic/shufa'
    broken_jpg_path = '../shufa_pic/broken_img'
    for jpg_file in tqdm(os.listdir(jpg_path)):
        src = os.path.join(jpg_path, jpg_file)
        try:
            image = Image.open(src)
        except UnidentifiedImageError:
            trg = os.path.join(broken_jpg_path, jpg_file)
            os.rename(src, trg)
            continue
