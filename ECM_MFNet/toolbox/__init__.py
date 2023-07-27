from .metrics import averageMeter, runningScore
from .log import get_logger
#from .optim import Ranger
#import torch_optimizer as optim
#from optim import Ranger
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['irseg', 'pst900']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')


def get_model(cfg):

    if cfg['model_name'] == 'ECM':
        from .models.ECM import ECM
        return ECM(n_classes=cfg['n_classes'])
