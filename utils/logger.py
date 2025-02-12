from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg

    def log_tensorboard(self, summary: SummaryWriter, log_dict: dict, ep, train=True):
        key_list = list(log_dict.keys())
        if train:
            cate = 'Train'
        else:
            cate = 'Test'

        print(f'-------------------[Writing a {cate} log on TensorBoard]')

        for key in key_list:
            summary.add_scalar(f'{cate}/{key}', log_dict[key], ep)

    def log_txt(self, log_dict, f, cate='Train'):
        key_list = list(log_dict)

        for key in key_list:
            f.write(f"{cate} [epoch: {log_dict[key]}\t{key}: ]")
