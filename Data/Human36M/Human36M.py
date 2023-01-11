from config import cfg
from Data.Human36M.h36m_dataset import Human36mDataset
from Data.Human36M.camera import *
import random
from common.visualization import render_animation
import cv2

annot_2d_path = 'Data/Human36M/data_2d_h36m_' + cfg.keypoints_data + '.npz'
annot_3d_path = 'Data/Human36M/data_3d_h36m.npz'
dataset = Human36mDataset(annot_3d_path)
print('Loading 3D detections...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                pos_3d = pos_3d * 1000
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load(annot_2d_path, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[
            subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                 subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            cam = dataset.cameras()[subject][cam_idx]
            '''
            if cfg.viz_subject == subject and cfg.viz_action == action:
                cfg.viz_video = 'Data/Human36M/videos/' + cfg.viz_subject + '/Videos/' + cfg.viz_action + '.' + cam[
                    'id'] + '.mp4'
                capture = cv2.VideoCapture( cfg.viz_video)
                while capture.isOpened():
                    run, frame = capture.read()
                    cvimg = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
                    for j in range(17):
                        pos = kps[0][j]
                        cv2.circle(cvimg, (int(pos[0]), int(pos[1])), 1, (0, 0, 255), 2)
                        cv2.imshow('Test Viewer', cvimg)
                        if cv2.waitKey(30) & 0xFF == ord('q'):
                         break
            '''
            # Normalize camera frame
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return predicted_aligned  # np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

class Human36M:
    def __init__(self, data_split):
        self.data_split = data_split
        self.annot_2d_path = annot_2d_path
        self.annot_3d_path = annot_3d_path
        self.subject_train = cfg.subjects_train
        self.subject_test = cfg.subjects_test
        self.stride = cfg.stride
        self.joint_num = 17
        self.joints_name = ( 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.root_idx = self.joints_name.index('Pelvis')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        #self.action_name_detail = ['Directions 1', 'Directions', 'Discussion 1', 'Discussion 2', 'Eating 1', 'Eating', 'Greeting 1', 'Greeting', 'Phoning 1', 'Phoning', 'Photo 1', 'Photo', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1', 'Sitting', 'SittingDown 1', 'SittingDown', 'Smoking 1', 'Smoking', 'Waiting 1', 'Waiting', 'WalkDog 1', 'WalkDog', 'Walking 1', 'Walking', 'WalkTogether 1', 'WalkTogether']
        self.action_name_detail = ['Directions 1', 'Discussion 1', 'Discussion 2', 'Eating 1', 'Eating', 'Greeting 2', 'Greeting', 'Phoning 2', 'Phoning 3', 'Photo 1', 'Photo', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1', 'Sitting', 'SittingDown 1', 'SittingDown', 'Smoking 2', 'Smoking', 'Waiting 1', 'Waiting', 'WalkDog 1', 'WalkDog', 'Walking 1', 'Walking', 'WalkTogether 1', 'WalkTogether']
        self.data = self.load_data()

    def load_data(self):
        if self.data_split == 'train':
            subjects_list = self.subject_train
            print('Train data load...')
        elif self.data_split == 'test':
            subjects_list = self.subject_test
            print('Test data load...')

        data = []
        frame_idx = 0
        for subject in subjects_list:
            for action in keypoints[subject].keys():
                poses_2d = keypoints[subject][action]
                poses_3d = dataset[subject][action]['positions_3d']
                cams = dataset.cameras()[subject]
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_2d)):  # Iterate across cameras
                    for j in range(len(poses_2d[i])):
                        if frame_idx % self.stride != 0:
                            frame_idx += 1
                            continue
                        frame_idx += 1
                        joint_img = poses_2d[i][j][:, :2].copy()
                        joint_cam = poses_3d[i][j]
                        img_width = cams[i]['res_w']
                        img_height = cams[i]['res_h']
                        f = cams[i]['focal_length']
                        c = cams[i]['center']

                        if self.data_split == 'train':
                            do_flip = random.random() <= 0.5
                            if do_flip:
                                joint_img[:, 0] *= -1
                                joint_img[kps_left + kps_right, :] = joint_img[kps_right + kps_left, :]
                                joint_cam[:, 0] *= -1
                                joint_cam[kps_left + kps_right, :] = joint_cam[kps_right + kps_left, :]
                        data.append({
                            'subject': subject,
                            'action': action,
                            'joint_img': joint_img,
                            'joint_cam': joint_cam,
                            'img_width': img_width,
                            'img_height': img_height,
                            'f': f,
                            'c': c,
                            'camera': i
                        })
        return data

    def evaluate(self, preds, result_dir, logger):
        print('Evaluation start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)

        error = np.zeros((sample_num, 17, 3))  # MPJPE
        error_p = np.zeros((sample_num, 17, 3))  # P-MPJPE
        error_action = [[] for _ in range(len(self.action_name))]  # MPJPE for each action
        error_action_p = [[] for _ in range(len(self.action_name))]  # P-MPJPE for each action
        for n in range(sample_num):
            gt = gts[n]
            gt_joint = gt['joint_cam']
            gt_root = gt['joint_cam'][0]
            action = gt['action']
            action_index = self.action_name.index(action.split(' ')[0])
            pred_root = preds[n]
            # error calculate
            gt_joint[0] = 0
            pred_root[0] = 0
            error[n] = (pred_root - gt_joint) ** 2
            pred_root_p = p_mpjpe(pred_root[None, ...], gt_joint[None, ...])
            error_p[n] = (pred_root_p[0] - gt_joint) ** 2
            error_action[action_index].append(error[n].copy())
            error_action_p[action_index].append(error_p[n].copy())

        # Total Error Calculation
        tot_err = np.mean(np.sqrt(np.sum(error, axis=2)))
        x_err = np.mean(np.sqrt(error[:, :, 0]))
        y_err = np.mean(np.sqrt(error[:, :, 1]))
        z_err = np.mean(np.sqrt(error[:, :, 2]))

        tot_err_p = np.mean(np.sqrt(np.sum(error_p, axis=2)))
        x_err_p = np.mean(np.sqrt(error_p[:, :, 0]))
        y_err_p = np.mean(np.sqrt(error_p[:, :, 1]))
        z_err_p = np.mean(np.sqrt(error_p[:, :, 2]))

        eval_summary = 'MPJPE >> tot: %.2f, x: %.2f, y: %.2f, z: %.2f\n' % (tot_err, x_err, y_err, z_err)
        eval_summary = eval_summary + 'P-MPJPE >> tot: %.2f, x: %.2f, y: %.2f, z: %.2f\n' % (
            tot_err_p, x_err_p, y_err_p, z_err_p)

        # error for each action
        for i in range(len(error_action)):
            err = np.array(error_action[i])
            err = np.mean(np.power(np.sum(err, axis=2), 0.5))
            err_p = np.array(error_action_p[i])
            err_p = np.mean(np.power(np.sum(err_p, axis=2), 0.5))
            eval_summary += (self.action_name[i] + ': %.2f (MPJPE) %.2f (P-MPJPE)\n' % (err, err_p))
        logger.info(eval_summary)

        return error

    def visualization(self, preds, result_dir, logger):
        print('Visualization start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)
        gt_save = [[] for _ in range(len(self.action_name_detail))]
        input_2d_save = [[] for _ in range(len(self.action_name_detail))]
        pred_save = [[] for _ in range(len(self.action_name_detail))]
        for n in range(sample_num):
            gt = gts[n]
            gt_joint = gt['joint_cam']
            gt_joint[0] = 0

            action = gt['action']
            camera = gt['camera']
            subject = gt['subject']
            if cfg.viz_camera == camera and cfg.viz_subject == subject:
                action_index = self.action_name_detail.index(action)
                gt_save[action_index].append(gt_joint.copy())
                input_2d_save[action_index].append(gt['joint_img'].copy())
                pred_save[action_index].append(preds[n].copy())

        for i in range(len(gt_save)):
            ground_truth = np.asarray(gt_save[i], dtype=np.float32)
            prediction = np.asarray(pred_save[i], dtype=np.float32)
            input_keypoints = np.asarray(input_2d_save[i], dtype=np.float32)

            cam = dataset.cameras()[cfg.viz_subject][cfg.viz_camera]
            prediction = camera_to_world(prediction / 1000, R=cam['orientation'], t=0)
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            ground_truth = camera_to_world(ground_truth / 1000, R=cam['orientation'], t=0)
            ground_truth[:, :, 2] -= np.min(ground_truth[:, :, 2])
            anim_output = {'MM-GCN': prediction, 'Ground truth': ground_truth}
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
            cfg.viz_video = 'Data/Human36M/videos/' + cfg.viz_subject + '/Videos/' + self.action_name_detail[i] + '.' + cam[
                'id'] + '.mp4'
            output_file = cfg.vis_dir + '/Results_' + cfg.viz_subject + '_' + self.action_name_detail[i] + '_' + str(
                cfg.viz_camera) + '.mp4'
            render_animation(input_keypoints, anim_output, dataset.skeleton(), dataset.fps(), cfg.viz_bitrate,
                             cam['azimuth'],
                             output_file, limit=cfg.viz_limit, downsample=cfg.viz_downsample, size=cfg.viz_size,
                             input_video_path=cfg.viz_video, viewport=(cam['res_w'], cam['res_h']),
                             input_video_skip=cfg.viz_skip)