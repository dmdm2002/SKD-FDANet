import os
import gc
import time
import torch
from numba import cuda

# fkd_lambda = [0.05, 0.1, 0.2, 0.3]
# cls_lambda = [0.95, 0.9, 0.8, 0.7]

# lambdas = [[0.3, 0.7, 0.3], [0.2, 0.8, 0.2], [0.4, 0.6, 0.4], [0.1, 0.9, 0.1]]

lambdas = [[0.3, 0.7, 0.3]]
db = ['Warsaw']
folds = ['2-fold']
models = ['FDADenseNetEncoder'] # DESADenseNetEncoder DenseNetEncoder FDESADenseNetEncoder

for db_name in db:
    for model in models:
        for _, (fkd_lambda, cls_lambda, mmd_lambda) in enumerate(lambdas):
            for fold in folds:
                device = cuda.get_current_device()
                device.reset()

                torch.cuda.empty_cache()
                gc.collect()

                path = f'E:/backup/Proposed/SKD-FDANet/stem_dense1-4_KD_lr_1e-4_to_1e-5/fkd0.3_cls0.7_mmd0.3/Train_PGLAV-GAN/{fold}/'

                os.system(f'python ./val_main.py --path={path} --model={model} --fkd_lambda={fkd_lambda} --cls_lambda={cls_lambda} --mmd_lambda={mmd_lambda} --dataset={db_name} --fold={fold}')

                time.sleep(3)
