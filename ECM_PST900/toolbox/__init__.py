from .metrics import averageMeter, runningScore
from .log import get_logger
#from .optim import Ranger
#import torch_optimizer as optim
#from optim import Ranger
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'nyuv2_new', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', 'irseg_msv']

    if cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PST900
        return PST900(cfg, mode='train'), PST900(cfg, mode='test'), PST900(cfg, mode='test')

def get_model(cfg):

    if cfg['model_name'] == 'ECM':
        from .models.ECM import ECM
        return ECM(n_classes=cfg['n_classes'])
