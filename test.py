import numpy as np
from config import cfg
import torch
from common.base import Tester
import torch.backends.cudnn as cudnn
from Data.Human36M.Human36M import dataset
from common.graph_utils import adj_mx_from_skeleton

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

    preds = []
    with torch.no_grad():
        for itr, (keypoint_img, joint_cam, alpha, subject, action) in enumerate(tester.batch_generator):
            if torch.cuda.is_available():
                keypoint_img = keypoint_img.cuda()
                if cfg.root_mode:
                    alpha = alpha.cuda()
            coord_pose = model_pose(keypoint_img, alpha)
            coord_pose = coord_pose.cpu().numpy()
            if cfg.root_mode is False:
                coord_pose[:, :1] = 0
            coord_out = coord_pose
            preds.append(coord_out)

    preds = np.concatenate(preds, axis=0)

    if cfg.visualization:
        tester._visualize(preds, cfg.result_dir)
    else:
        tester._evaluate(preds, cfg.result_dir)

if __name__ == "__main__":
    main()
