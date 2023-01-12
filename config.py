import os
import errno
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    # General arguments
    parser.add_argument('-d', '--dataset', default='Human36M', type=str, metavar='NAME',
                        help='Training dataset')  # Human36M, MuCo, PW3D
    parser.add_argument('-k', '--keypoint', default='gt', type=str, metavar='NAME',
                        help='keypoints dataset')  # gt, CPN, Detectron, sh
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST',
                        help='test subjects separated by comma')
    parser.add_argument('-p', '--protocol', default=2, type=int, metavar='N', help='Protocol for H36M')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('--test_model', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    # Model arguments
    parser.add_argument('-ch', '--channels', default=128, type=int, metavar='N', help='number of channels in convolution layers')
    parser.add_argument('-sk', '--skeleton_graph', default=1, type=int, metavar='N', help='distance of skeleton graph')
    parser.add_argument('-j_in', '--joint_in', default=17, type=int, metavar='N', help='number of input joints')
    parser.add_argument('-f_in', '--feature_in', default=2, type=int, metavar='N', help='number of input features')
    parser.add_argument('-j_out', '--joint_out', default=1, type=int, metavar='N', help='number of output joints')
    parser.add_argument('-n', '--num_layers', default=4, type=int, metavar='N', help='number of layers')
    parser.add_argument('-drop', '--drop_out', default=None, type=float, metavar='Drop', help='drop-out prob')
    parser.add_argument('-nlocal', '--Non_Local', action='store_true', help='use non local block')
    parser.add_argument('-NoAM', '--NoAffinityModulation', action='store_true', help='No Affinity Modulation')

    # Training arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=5, type=float, metavar='LR',
                        help='learning rate decay factor per epoch')
    # GPU arguments
    parser.add_argument('-g', '--gpus', default='0', type=str, metavar='N', help='gpu index')
    parser.add_argument('-th', '--thread', default=8, type=int, metavar='N', help='number of worker')

    # Visualization
    parser.add_argument('--visualization',  action='store_true', help='visualization')
    parser.add_argument('--viz_subject', type=str, default='S11', metavar='STR', help='subject to render')
    parser.add_argument('--viz_action', type=str, default='Directions', metavar='STR', help='action to render')
    parser.add_argument('--viz_camera', type=int, default=1, metavar='N', help='camera to render')
    parser.add_argument('--viz_video', type=str, default=None, metavar='PATH', help='path to input video')
    parser.add_argument('--viz_skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz_output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz_bitrate', type=int, default=1000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz_limit', type=int, default=500, metavar='N', help='only render first N frames')
    parser.add_argument('--viz_downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz_size', type=int, default=5, metavar='N', help='image size')

    args = parser.parse_args()

    return args


def check_directory(path):
    try:
        os.makedirs(path)  # Create checkpoint directory if it does not exist
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', path)


class Config:
    args = parse_args()

    num_joints_in = args.joint_in
    in_features = args.feature_in
    num_joints_out = args.joint_out
    num_layers = args.num_layers
    data_set = args.dataset
    keypoints_data = args.keypoint
    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')
    protocol = args.protocol
    dataset_path = 'Data/dataset/'
    output_dir = args.checkpoint + '/'
    model_dir = output_dir + 'model'
    vis_dir = output_dir + 'vis'
    log_dir = output_dir + 'log'
    result_dir = output_dir + 'result'
    test_model = args.test_model

    ## model setting
    channels = args.channels
    skeleton_graph = args.skeleton_graph
    dropout = args.drop_out
    Non_Local = args.Non_Local
    NoAffinityModulation = args.NoAffinityModulation

    ## training config
    stride = args.stride
    lr_dec_epoch = [5]
    end_epoch = args.epochs
    lr = args.learning_rate
    lr_dec_factor = args.lr_decay
    batch_size = args.batch_size

    ## testing config
    test_batch_size = args.batch_size

    ## Visualization
    visualization = args.visualization
    viz_subject = args.viz_subject
    viz_action = args.viz_action
    viz_camera = args.viz_camera
    viz_video = args.viz_video
    viz_skip = args.viz_skip
    viz_output = args.viz_output
    viz_bitrate = args.viz_bitrate
    viz_limit = args.viz_limit
    viz_downsample = args.viz_downsample
    viz_size = args.viz_size

    ## others
    num_thread = args.thread
    gpu_ids = args.gpus
    num_gpus = 1
    resume = args.resume
    continue_train = False
    if resume is not None:
        continue_train = True

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

    check_directory(args.checkpoint)
    check_directory(model_dir)
    check_directory(vis_dir)
    check_directory(log_dir)
    check_directory(result_dir)

cfg = Config()

cfg.set_args(cfg.gpu_ids, cfg.continue_train)
