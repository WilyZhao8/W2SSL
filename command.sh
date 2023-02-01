CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py --moco-m-cos --crop-min=.2 --dist-url 'tcp://localhost:10109' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --batch-size 512 --epochs 200 --loss_crop 1.0 /media/wily/SSD/ImageNet_ILSVRC2012
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_lincls.py -a resnet50 --batch-size 256 --epochs 100  --dist-url 'tcp://localhost:10016'--multiprocessing-distributed --world-size 1 --rank 0 --print-freq 100 --pretrained checkpoint_0199.pth.tar --resume lin_checkpoint.pth.tar /media/wily/SSD/ImageNet_ILSVRC2012




#python main_lego_mocov3.py --moco-m-cos --crop-min=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --resume checkpoint_0013.pth.tar /media/user/sdisk/Imagenet2012/
