import re

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import random
random.seed(1004)


def hide_and_swap(image_path, target_path, grid_size=(4, 4), hide_ratio=0.8, random_ratio=True):
    if random_ratio:
        hide_ratio = round(random.uniform(.5, hide_ratio), 2)
    # 이미지 열기
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 포맷이므로 RGB로 변환

    # 대상 이미지 열기
    target_image = cv2.imread(target_path)
    # target_image = cv2.resize(target_image, (image.shape[1], image.shape[0]))  # 크기를 원본 이미지와 맞추기
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # 이미지 크기
    h, w, _ = image.shape

    # 그리드 크기
    grid_h, grid_w = grid_size

    # 각 패치의 크기
    patch_h = h // grid_h
    patch_w = w // grid_w

    # 전체 패치 리스트 생성
    patches = [(i, j) for i in range(grid_h) for j in range(grid_w)]

    # 마스킹할 패치의 개수 계산
    num_patches_to_hide = int(hide_ratio * len(patches))

    # 무작위로 패치를 선택하여 마스킹
    patches_to_hide = random.sample(patches, num_patches_to_hide)
    for (i, j) in patches_to_hide:
        top = i * patch_h
        left = j * patch_w
        image[top:top + patch_h, left:left + patch_w] = target_image[top:top + patch_h, left:left + patch_w]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, hide_ratio


if __name__ == '__main__':
    real_root = 'M:/2nd/dataset/Warsaw/original/A'
    fake_root = 'E:/dataset/Ocular/Warsaw/CycleGAN/2-fold/A/fake'

    real_images = glob.glob(f'{real_root}/*/*')
    print(real_images)

    output_path = 'E:/dataset/Ocular/Warsaw/RandomInjections/CycleGAN-RandomPatchInjectionAug/2-fold/A/fake'
    os.makedirs(output_path, exist_ok=True)

    for idx, real_path in enumerate(real_images):
        name = real_path.split('\\')[-1]
        name = re.compile('.png').sub('_A2B.png', name)
        fake_path = f'{fake_root}/{name}'

        swapped_image, hide_ratio = hide_and_swap(real_path, fake_path, grid_size=(32, 32), hide_ratio=0.8)
        cv2.imwrite(f'{output_path}/{name}', swapped_image)

        print(f'[{idx}/{len(real_images)}] ratio: {hide_ratio}')
