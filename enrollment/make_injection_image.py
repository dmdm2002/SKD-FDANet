import glob
import os

import cv2
import random
import tqdm
random.seed(1004)


def injection_augmentation(img_r, img_f, img_size=224, ratio=.7, random_ratio=True):
    assert img_size * ratio < img_size, "Can't crop an area larger than or equal to the image."

    if random_ratio:
        random_ratio = round(random.uniform(.3, ratio), 2)
        crop_w = round(img_size * random_ratio)

        random_ratio = round(random.uniform(.3, ratio), 2)
        crop_h = round(img_size * random_ratio)
    else:
        random_ratio = ratio
        crop_w = round(img_size * ratio)
        crop_h = round(img_size * ratio)

    mid_x, mid_y = img_size // 2, img_size // 2
    offset_x, offset_y = crop_w // 2, crop_h // 2

    injection_img = img_r.copy()
    injection_img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x] = img_f[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]

    return injection_img, random_ratio


if __name__ == '__main__':
    real_root = 'M:/2nd/dataset/Warsaw/original/B'
    fake_root = 'M:/3rd/dataset/Ocular/Warsaw/PGLAV-GAN/1-fold/B/fake'

    real_images = glob.glob(f'{real_root}/*/*')
    print(real_images)

    output_path = 'M:/3rd/dataset/Ocular/Warsaw/PGLAV-GAN-InjectionAug/1-fold/B/fake'
    os.makedirs(output_path, exist_ok=True)

    for idx, real_path in enumerate(real_images):
        name = real_path.split('\\')[-1]
        fake_path = f'{fake_root}/{name}'

        img_r = cv2.imread(real_path)
        img_r = cv2.resize(img_r, (224, 224))
        img_f = cv2.imread(fake_path)

        injection_img, ratio = injection_augmentation(img_r, img_f, img_size=224, ratio=.7, random_ratio=True)
        cv2.imwrite(f'{output_path}/{name}', injection_img)

        print(f'[{idx}/{len(real_images)}] ratio: {ratio}')

