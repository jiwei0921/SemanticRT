import os
import time
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt, save_ckpt

from toolbox.datasets.irseg import IRSeg
from toolbox.datasets.pst900 import PST900
from toolbox.datasets.semanticRT import SemanticRT

def evaluate(logdir, save_predict=False, options=None, prefix=''):

    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda:0')
    # Note: During inference, you need to ensure that the device id is aligned with "./toolbox/utils.py/line-201"

    loaders = []
    for opt in options:
        #dataset = IRSeg(cfg, mode=opt)
        # dataset = PST900(cfg, mode=opt)
        dataset = SemanticRT(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
        cmap = dataset.cmap

    model = get_model(cfg).to(device)

    model = load_ckpt(logdir, model, prefix=prefix)

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):

                time_start = time.time()

                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    [d, e] = model(image, depth)[-1]
                    predict = d

                predict = predict.max(1)[1].cpu().numpy()
                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)

                time_meter.update(time.time() - time_start, n=image.size(0))

                if save_predict:
                    predict = predict.squeeze(0)
                    predict = class_to_RGB(predict, N=len(cmap), cmap=cmap)
                    predict = Image.fromarray(predict)
                    predict.save(os.path.join(save_path, sample['label_path'][0]+'.png'))

        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        for k, v in metrics[0].items():
            print(k, f'{v:.4f}')

        print('iou for each class .....')
        for k, v in metrics[1].items():
            print(k, f'{v:.4f}')
        print('acc for each class .....')
        for k, v in metrics[2].items():
            print(k, f'{v:.4f}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", default="./run/", type=str,
                        help="run logdir")
    parser.add_argument("-s", type=bool, default=True,
                        help="save predict or not")
    args = parser.parse_args()


    evaluate(args.logdir, save_predict=args.s, options=['test'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=[ 'test', 'test_day','test_night', 'test_mc', 'test_mo', 'test_hard'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['val'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_day'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_night'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_mc'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_mo'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_hard'], prefix='')

