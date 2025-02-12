import os
import gc
import time
import torch
from numba import cuda

# fkd_lambda = [0.05, 0.1, 0.2, 0.3]
# cls_lambda = [0.95, 0.9, 0.8, 0.7]

lambdas = [[0.3, 0.7, 0.3]]
db = ['Warsaw']
folds = ['1-fold', '2-fold']
models = ['FDADenseNetEncoder'] # DESAResNetEncoder DESADenseNetEncoder

for db_name in db:
    for model in models:
        for _, (kld_lambda, cls_lambda, mmd_lambda) in enumerate(lambdas):
            for fold in folds:
                device = cuda.get_current_device()
                device.reset()

                torch.cuda.empty_cache()
                gc.collect()

                path = f'E:/backup/Proposed/SKD-FDANet/{db_name}/NewWarmUp/stem_dense1-4_KD_lr_1e-4_to_1e-5/kld{kld_lambda}_cls{cls_lambda}_mmd{mmd_lambda}/Train_PGLAV-GAN/{fold}/'
                os.system(f'python ./main.py --path={path} --model={model} --kld_lambda={kld_lambda} --cls_lambda={cls_lambda} --mmd_lambda={mmd_lambda} --dataset={db_name} --fold={fold}')

                time.sleep(3)

