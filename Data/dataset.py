import copy
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

class DatasetLoader(Dataset):
    def __init__(self, db, is_train, dataset_name):
        self.db = db.data
        self.dataset_name = dataset_name
        self.joint_num = db.joint_num
        self.root_idx = db.root_idx
        self.is_train = is_train

    def __getitem__(self, index):
        dataset_name = self.dataset_name
        data = copy.deepcopy(self.db[index])
        keypoint_img = data['joint_img']
        joint_cam = data['joint_cam']
        subject = data['subject']
        action = data['action']

        if self.is_train:
            return keypoint_img, joint_cam
        else:
            return keypoint_img, joint_cam, subject, action

    def __len__(self):
        return len(self.db)

class PoseBuffer(Dataset):
    def __init__(self, poses_3d, poses_2d, score=None):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        print('Generating {} poses...'.format(self._poses_3d.shape[0]))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float() * 1000
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_2d, out_pose_3d

    def __len__(self):
        return len(self._poses_2d)
