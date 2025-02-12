import glob

from PIL import Image
import os
import tqdm

# checkdir = 'E:/dataset/Ocular/Warsaw/RandomInjections/PGLAV-GAN-RandomPatchInjectionAug/1-fold/B/fake'
checkdirs = [
    "E:/dataset/Ocular/Warsaw/original/B/*/*",
    # "E:/dataset/Ocular/Warsaw/original/A/*/*",
    'E:/dataset/Ocular/Warsaw/PGLAV-GAN/1-fold/B/fake/*',
    # 'E:/dataset/Ocular/Warsaw/PGLAV-GAN/2-fold/A/fake/*',
    'E:/dataset/Ocular/Warsaw/PGLAV-GAN-RandomPatchInjectionAug/1-fold/B/fake/*',
    # 'E:/dataset/Ocular/Warsaw/PGLAV-GAN-RandomPatchInjectionAug/2-fold/A/fake/*'
]

for checkdir in checkdirs:
    files = glob.glob(checkdir)
    # files = glob.glob('M:/2nd/dataset/Warsaw/original/B/*/*')

    # print(files)

    for _, file in enumerate(tqdm.tqdm(files)):
        try:
            image = Image.open(file).load()

        except Exception as e:
            print(f"An exception is raised: {e}")
            print(f'{file}')
