import argparse
import numpy as np


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--free_gpu_id',
                        type=int,
                        default=0,
                        help='The selected gpu.')

    parser.add_argument('--x_image_channel', type=int, default=2)
    parser.add_argument('--y_image_channel', type=int, default=2)
    parser.add_argument('--conv_kernel_size', type=int, default=3)
    parser.add_argument('--conv_padding_size', type=int, default=1)
    parser.add_argument('--lstm_dim', type=int, default=400)  # For mask_CNN model
    parser.add_argument('--fc1_dim', type=int, default=600)  # For mask_CNN model
    parser.add_argument('--sf', type=int, default=7, help='The spreading factor.')
    parser.add_argument('--bw', type=int, default=125000, help='The bandwidth.')
    parser.add_argument('--fs', type=int, default=1000000, help='The sampling rate.')
    parser.add_argument( '--server', action='store_true', default=False, help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument( '--normalization', action='store_true', default=False, help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument( '--init_zero_weights', action='store_true', default=False, help= 'Choose whether to initialize the generator conv weights to 0 (implements the identity function).')
    parser.add_argument( '--load_iters', type=int, default=100000, help= 'The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument( '--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (default 0.0003)') 
    parser.add_argument('--sorting_type', type=int, default=4, choices=[4], help='The index for the selected domain.')
    parser.add_argument('--scaling_for_imaging_loss', type=int, default=1, help='The scaling factor for the imaging loss')
    parser.add_argument('--scaling_for_classification_loss', type=int, default=8, help='The scaling factor for the classification loss')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--evaluations_dir', type=str, default='/data/djl/evaluations', help='Choose the root path to rf signals.')
    parser.add_argument('--log_dir', type=str, default='/data/djl/logs0404', help='path for log files.')
    parser.add_argument('--data_dir', type=str, default='/data/djl/data0306/data', help='Choose the root path to rf signals.')
    parser.add_argument('--feature_name', type=str, default='chirp', choices=['chirp'])
    parser.add_argument('--groundtruth_code', type=str, default='35', choices=['35', '50'])
    parser.add_argument("--code_list", nargs='+', default=[round(i, 1) for i in list(np.arange(0, 128, 0.1))], type=float)
    parser.add_argument("--snr_list", nargs='+', default=list(range(-16, -15)), type=int)  # for train: -25:0, test: -40, 16
    parser.add_argument( '--ratio_bt_train_and_test', type=float, default=0.9, help='The ratio between the train and the test dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='/data/djl/checkpoints/default')
    parser.add_argument('--load', type=str, default='yes')
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--test_step', type=int, default=500)
    parser.add_argument('--test_log_step', type=int, default=1000)
    parser.add_argument( '--train_iters', type=int, default=3000, help= 'The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--sample_every', type=int, default=10000)
    parser.add_argument('--checkpoint_every', type=int, default=5000)
    parser.add_argument('--load_checkpoint_dir', type=str, default='/data/djl')
    parser.add_argument('--stack_imgs', type=int, default=3)  # Multi-image SR
    parser.add_argument('--train_datacnt', type=int, default=500)
    parser.add_argument('--test_datacnt', type=int, default=5)
    parser.add_argument('--use_old_data', type=str, default='True')
    parser.add_argument('--write_new_data', type=str, default='False')

    return parser
