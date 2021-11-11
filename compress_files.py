from PIL import Image
import os

img_dir = "alex_2small_loops_images"
filenames_dir = sorted(next(os.walk(img_dir), (None, None, []))[2])

for i in filenames_dir:
    img_path = os.path.join(img_dir, i)
    im = Image.open(img_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(f"{img_path[:-3]}jpg")
    os.remove(img_path)