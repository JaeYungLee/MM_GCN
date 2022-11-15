import numpy as np
import time
from config import cfg
import torch
from common.base import Tester_Unified
from common.base import Tester
import torch.backends.cudnn as cudnn
from Data.Human36M.Human36M import dataset
from common.graph_utils import adj_mx_from_skeleton

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def main():
    cfg.set_args(cfg.gpu_ids, cfg.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True

    pose_model_file = cfg.test_model
    tester = Tester(pose_model_file)

    tester._make_batch_generator()
    adj, adj_ext_1, adj_ext_2, adj_ext_3, adj_ext_4 = adj_mx_from_skeleton(dataset.skeleton())
    tester._make_model(adj, adj_ext_1, adj_ext_2, adj_ext_3, adj_ext_4)
    model_pose = tester.model

    batch_time = AverageMeter()
    preds = []
    with torch.no_grad():
        for itr, (keypoint_img, joint_cam, alpha, subject, action) in enumerate(tester.batch_generator):
            if torch.cuda.is_available():
                keypoint_img = keypoint_img.cuda()
                if cfg.root_mode:
                    alpha = alpha.cuda()
            infer = time.time()
            coord_pose = model_pose(keypoint_img, alpha)
            batch_time.update(time.time() - infer)
            coord_pose = coord_pose.cpu().numpy()
            if cfg.root_mode is False:
                coord_pose[:, :1] = 0
            coord_out = coord_pose
            preds.append(coord_out)

    print('Average Inference Time (per Batch) : %.6f s' % batch_time.avg)
    preds = np.concatenate(preds, axis=0)
    if cfg.visualization:
        tester._visualize(preds, cfg.result_dir)
    else:
        tester._evaluate(preds, cfg.result_dir)

if __name__ == "__main__":
    main()