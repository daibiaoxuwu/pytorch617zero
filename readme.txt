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

....../pytorch70orig
return the model to original form

0509:
CUDA_VISIBLE_DEVICES=1 python3 main.py --snr_list -15 --train_iters 300000 --load yes --load_checkpoint_dir /data/djl/evaluations/mult12_ckpt_-21_-21_3 --load_iters 30000 --checkpoint_dir /data/djl/checkpoints/ckpt0509-15-1o --batch_size 4 --use_old_data False
20000epoch lr=0.0002 reach 97 accuracy snr=-15
batchsize increase to 16: train 500 epoch, reach 99.2

CUDA_VISIBLE_DEVICES=1 python3 main.py --snr_list -16 --train_iters 300000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0509-15-1o --load_iters 19999 --checkpoint_dir /data/djl/checkpoints/ckpt0509-15-2o --checkpoint_every 3000
snr-16: immediate 95% 

CUDA_VISIBLE_DEVICES=1 python3 main.py --snr_list -17 --train_iters 300000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0509-15-1o --load_iters 19999 --checkpoint_dir /data/djl/checkpoints/ckpt0509-15-2o --checkpoint_every 3000
snr-17: 84% - 3000epoch 0.9453125(batchsize 16)
snr-19: 75% - 3000epoch 0.82 --9000epoch - 0.84375 
snr-20: 0.6796875% - 5999 0.74%
snr-21: 0.58 - 8999 0.65



move to pytorch70trian
(base) dujialuo@danlu:~/NELoRa-Sensys/neural_enhanced_demodulation/pytorch70train$ CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 30000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0509-15-5o --load_iters 8999 --checkpoint_dir /data/djl/checkpoints/ckpt0509-15-6o --checkpoint_every 3000 --snr_list -23 load checkpoint loading model checkpoint from /data/djl/checkpoints/ckpt0509-15-5o/8999_C_XtoY.pkl /data/djl/checkpoints/ckpt0509-15-5o/8999_maskCNN.pkl start training with snr [-23] stack 3 load data max steps: 4000 dataset count of each of the 128  symbols: max 500 min 500 avg 500.0 load data max steps: 40 dataset count of each of the 128  symbols: max 5 min 5 avg 5.0 start new training epoch start testing.. test accuracy 0.321875 logged to /data/djl/logs0404/log-23_-23_3.txt Train Iteration [   100/30000] | G_Y_loss: 73.5872| G_Image_loss: 69.8121| G_Class_loss: 3.7751 | Time: 41.71
snr-23: 0.32 -14000epochs 0.50
CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 30000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0509-15-5o --load_iters 8999 --checkpoint_dir /data/djl/checkpoints/ckpt0509-15-6o --checkpoint_every 3000 --snr_list -23


CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 30000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0509-15-6o --load_iters 14999 --checkpoint_dir /data/djl/checkpoints/ckpt0509-15-7o --checkpoint_every 3000 --snr_list -24
snr=-24 no gain 0.37


retrain load from
CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 30000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0513-lky-2 --checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-1 --checkpoint_every 1000 --snr_list -15 --batch_size 4 --load_iters 29999

then finetune
 CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 1000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-1 --checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-2 --checkpoint_every 500 --snr_list -15 --batch_size 16 --load_iters 5999 --log_dir /data/djl/logs  1670  CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 2000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-2 --checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-3 --checkpoint_every 500 --snr_list -15 --batch_size 64 --load_iters 9999 --log_dir /data/djl/logs

CUDA_VISIBLE_DEVICES=1 python3 main.py --train_iters 2000 --load yes --load_checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-3 --checkpoint_dir /data/djl/checkpoints/ckpt0516-70train-4 --checkpoint_every 500 --snr_list -15 --batch_size 16 --load_iters 999 --log_dir /data/djl/logs --test_step 500
batchsize16 weight1:1 or 1:2 can achieve 0.97
