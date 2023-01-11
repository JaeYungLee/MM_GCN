import torch
import numpy as np
from config import cfg
import cv2

skeleton_topology_H36M = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],  # Right-Leg, Left-Leg
                          [0, 7], [7, 8], [8, 9], [9, 10],  # Center-Line
                          [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]  # Left-Arm, Right-Arm

skeleton_topology_H36M_p1 = [0, 1, 2, 0, 4, 5,  # Right-Leg, Left-Leg
                             0, 7, 8, 9,  # Center-Line
                             8, 11, 12, 8, 14, 15]  # Left-Arm, Right-Arm

skeleton_topology_H36M_p2 = [1, 2, 3, 4, 5, 6,  # Right-Leg, Left-Leg
                             7, 8, 9, 10,  # Center-Line
                             11, 12, 13, 14, 15, 16]  # Left-Arm, Right-Arm

def mpjpe(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def mae(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.sum(torch.abs(predicted - target), dim=len(target.shape) - 1))

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def compute_PCK(gts, preds, scales=1000, eval_joints=None, threshold=150):
    PCK_THRESHOLD = threshold
    sample_num = len(gts)
    total = 0
    true_positive = 0
    if eval_joints is None:
        eval_joints = list(range(gts.shape[1]))

    for n in range(sample_num):
        gt = gts[n]
        pred = preds[n]
        # scale = scales[n]
        scale = 1
        per_joint_error = np.take(np.sqrt(np.sum(np.power(pred - gt, 2), 1)) * scale, eval_joints, axis=0)
        true_positive += (per_joint_error < PCK_THRESHOLD).sum()
        total += per_joint_error.size

    pck = float(true_positive / total) * 100
    return pck

def compute_AUC(gts, preds, scales=1, eval_joints=None):
    # This range of thresholds mimics 'mpii_compute_3d_pck.m', which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = np.linspace(0, 150, 31)
    pck_list = []
    for threshold in thresholds:
        pck_list.append(compute_PCK(gts, preds, scales, eval_joints, threshold))

    auc = np.mean(pck_list)

    return auc

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord

def get_bbox(joint_img):
    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:, 0])
    ymin = np.min(joint_img[:, 1])
    xmax = np.max(joint_img[:, 0])
    ymax = np.max(joint_img[:, 1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1

    bbox[0] = (xmin + xmax) / 2. - width / 2 * 1.2
    bbox[1] = (ymin + ymax) / 2. - height / 2 * 1.2
    bbox[2] = width * 1.2
    bbox[3] = height * 1.2

    return bbox

def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = cfg.input_shape[1] / cfg.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)

def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped
