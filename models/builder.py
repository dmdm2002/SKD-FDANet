from models.DenseNet.fda_densenet_encoder import FDADensenetEncoder
from models.DenseNet.desa_densenet_encoder import DESADensenetEncoder
from models.DenseNet.dsa_densenet_encoder import DSADensenetEncoder
from models.DenseNet.esa_densenet_encoder import ESADensenetEncoder
from models.DenseNet.sa_densenet_encoder import SADensenetEncoder


def model_builder(cfg: dict):
    # ---------------------------------------------------------------------------
    # ------------------------------- Build Model -------------------------------
    # ---------------------------------------------------------------------------
    if cfg['model'] == 'FDADenseNetEncoder':
        return FDADensenetEncoder(cfg['enc_model'], cfg['cls_num']).to(cfg['device'])

    elif cfg['model'] == 'DESADenseNetEncoder':
        return DESADensenetEncoder(cfg['enc_model'], cfg['cls_num']).to(cfg['device'])

    elif cfg['model'] == 'DSADenseNetEncoder':
        return DSADensenetEncoder(cfg['enc_model'], cfg['cls_num']).to(cfg['device'])

    elif cfg['model'] == 'ESADenseNetEncoder':
        return ESADensenetEncoder(cfg['enc_model'], cfg['cls_num']).to(cfg['device'])

    elif cfg['model'] == 'SADenseNetEncoder':
        return SADensenetEncoder(cfg['enc_model'], cfg['cls_num']).to(cfg['device'])

    else:
        print('지원하지 않는 모델입니다.')
        exit()
