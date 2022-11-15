'''
# This source code is written by Lee Jae Yung
# Some source code is borrowed from Modulate-GCN / VideoPose3D
# E-Mail : ljy321456@gmail.com
'''

import torch
import torch.backends.cudnn as cudnn
import os
import os.path as path

from config import cfg
from Data.Human36M.Human36M import dataset
from common.graph_utils import adj_mx_from_skeleton
from common.base import Trainer
from common.base import Tester
from common.utils import mpjpe
from common.utils import mae

def main():
    cudnn.fastest = True
    cudnn.benchmark = True

    # affinity matrix for multi-hop relationship
    adj, adj_ext_1, adj_ext_2, adj_ext_3, adj_ext_4 = adj_mx_from_skeleton(dataset.skeleton())

    # Trainer
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model(adj, adj_ext_1, adj_ext_2, adj_ext_3, adj_ext_4)

    # Tester
    cfg.stride = 1
    tester = Tester(test_model=None, is_train=True)
    tester._make_batch_generator()
    tester._make_model(adj, adj_ext_1, adj_ext_2, adj_ext_3, adj_ext_4)
    cur_lr = cfg.lr
    start_epoch = trainer.start_epoch
    if cfg.continue_train is True:
        file_path = path.join(cfg.model_dir, cfg.resume)
        print("==> Loading checkpoint '{}'".format(file_path))
        if path.isfile(file_path):
            ckpt = torch.load(file_path)
            start_epoch = ckpt['epoch'] + 1
            cur_lr = ckpt['lr']
            trainer.model.load_state_dict(ckpt['network'])
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
            for g in trainer.optimizer.param_groups:
                g['lr'] = cur_lr

    losses_3d_train = []
    losses_3d_eval = []
    for epoch in range(start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch, cur_lr)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        # Training
        epoch_loss = 0
        for itr, (keypoint_img, joint_cam, alpha) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            if torch.cuda.is_available():
                keypoint_img = keypoint_img.cuda()
                joint_cam = joint_cam.cuda()
                if cfg.root_mode:
                    alpha = alpha.cuda()
            # forward
            trainer.optimizer.zero_grad()
            pred = trainer.model(keypoint_img, alpha)

            if cfg.root_mode:
                loss_L2 = mpjpe(pred, joint_cam[:, :1])
                loss_L1 = mae(pred, joint_cam[:, :1])
            else:
                joint_cam[:, :1] = 0
                loss_L2 = mpjpe(pred, joint_cam)
                loss_L1 = mae(pred, joint_cam)
            loss = 0.9 * loss_L2 + 0.1 * loss_L1

            loss.backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            epoch_loss = epoch_loss + loss_L2.detach()
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        epoch_loss = epoch_loss / trainer.itr_per_epoch
        losses_3d_train.append(epoch_loss)
        cur_lr = trainer.get_lr()

        # Evaluation
        epoch_eval_loss = 0
        with torch.no_grad():
            tester.model.load_state_dict(trainer.model.state_dict())
            tester.model.eval()
            for itr, (keypoint_img, joint_cam, alpha, subject, action) in enumerate(tester.batch_generator):
                if torch.cuda.is_available():
                    keypoint_img = keypoint_img.cuda()
                    joint_cam = joint_cam.cuda()
                coord_out = tester.model(keypoint_img, alpha)
                if cfg.root_mode:
                    loss_eval = mpjpe(coord_out, joint_cam[:, :1])
                else:
                    joint_cam[:, :1] = 0
                    coord_out[:, :1] = 0
                    loss_eval = mpjpe(coord_out, joint_cam)
                epoch_eval_loss = epoch_eval_loss + loss_eval.detach()
            epoch_eval_loss = epoch_eval_loss / tester.itr_per_epoch
            losses_3d_eval.append(epoch_eval_loss)

        # Results display
        screen = [
            'Epoch %d/%d:' % (epoch, cfg.end_epoch),
            'lr: %g' % (trainer.get_lr()),
            '%.4fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
            '%s: %.4f' % ('loss_coord', epoch_loss.detach()),
            '%s: %.4f' % ('eval_loss', epoch_eval_loss.detach()),
        ]
        trainer.logger.info(' '.join(screen))

        # Save model
        if cfg.keypoints_data != 'gt':
            save_threshold = 52
        else:
            save_threshold = 39
        error = epoch_eval_loss.cpu().numpy()
        file_name = 'epoch_' + str(epoch) + '_' + str(error) + '_model.pth.tar'
        if error < save_threshold:
            trainer.save_model({
                'epoch': epoch,
                'lr': cur_lr,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, file_name)


if __name__ == '__main__':
    main()