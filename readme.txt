from the original checkpoint file:
python3 main.py --dir_comment sf7_125k --batch_size 8 --root_path . --data_dir ../../sf7_125k --groundtruth_code 35 --normalization --train_iter 100000 --ratio_bt_train_and_test 0.8 --network end2end --load yes
train on -5 snr for 100000 iterations then (data_dir:sf7_125k) saved at evaluations/sf7_125k_checkpoints_5
then
python3 main.py --dir_comment sf7_125k --batch_size 8 --root_path . --data_dir ../../sf7_125k_old --groundtruth_code 35 --normalization --train_iter 1000 --ratio_bt_train_and_test 0.8 --network end2end --load yes --load_iters 100000 --log_step 10


ls -al evaluations/sf7_125k_checkpoints/
cp -r evaluations/sf7_125k_checkpoints/ evaluations/sf7_125k_checkpoints_5/
python3 main.py --dir_comment sf7_125k --batch_size 8 --root_path . --data_dir ../../sf7_125k_old --groundtruth_code 35 --normalization --train_iter 1000 --ratio_bt_train_and_test 0.8 --network end2end --load yes --load_iters 100000 --log_step 10

utils: filter all split[1] != 35 and split[1] != -4 (-4 not sent completely)
use sf7_125k_100
-35~-10,35 split:[0~7] [1]=35:[7]=0
data_file_name[1] = self.groundtruth_code  
data_file_name[-1] = '0.mat'


pytorch70
CUDA_VISIBLE_DEVICES=1 python3 main.py --snr_list -22 --train_iters 30000 --load yes --load_iters 15999 --load_checkpoint_dir /data/djl/checkpoints/ckpt0508-22-2 --checkpoint_dir /data/djl/checkpoints/ckpt0508-22-3
SNR-22: max 0.4828125
    parser.add_argument('--train_datacnt', type=int, default=500)
    50 can only last 400 epochs at batchsize 16
