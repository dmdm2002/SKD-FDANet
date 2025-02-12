import yaml
import os


def get_configs(path):
    assert os.path.exists(path), f"경로[{path}]에 해당 파일이 존재하지 않습니다."
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def save_configs(cfg):
    with open(f"{cfg['log_path']}/train_parameters.yml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def cal_metrics(tp, tn, fp, fn):
    acc = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) != 0 else 0
    apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
    bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0
    acer = (apcer + bpcer) / 2

    return acc, apcer, bpcer, acer
