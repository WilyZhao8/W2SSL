echo "start to run......"
#CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url 'tcp://localhost:10126' --multiprocessing-distributed --world-size 1 --rank 0 -a resnet50 --lr 0.03 --batch-size 256 --epoch 200 --save-dir outputs/jigclu_pretrain/  --loss-t 0.3 --cross-ratio 0.3 /media/wily/SSD/data/ImageNet100
#CUDA_VISIBLE_DEVICES=0,1 python main_lincls.py -a resnet50 --batch-size 256 --epochs 100 --resume lin_checkpoint.pth.tar  --dist-url 'tcp://localhost:10001'--multiprocessing-distributed --world-size 1 --rank 0 --print-freq 100 --pretrained checkpoint_0039.pth.tar  /media/wily/SSD/data/ImageNet100/
wait
#CUDA_VISIBLE_DEVICES=0,1 python main_lincls.py --dist-url 'tcp://localhost:10026' --multiprocessing-distributed --world-size 1 --rank 0 -a resnet50 --lr 1.0 --batch-size 256 --prefix module.encoder. --pretrained outputs/jigclu_pretrain/model_best.pth.tar --save-dir outputs/jigclu_linear/ /media/wily/SSD/data/ImageNet100
#CUDA_VISIBLE_DEVICES=0,1 python main_crop_moco.py --moco-m-cos --crop-min=.2 --dist-url 'tcp://localhost:10109' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --batch-size 256 --epochs 200 --loss_crop 1.0  /media/wily/SSD/ImageNet_ILSVRC2012/
#wait
CUDA_VISIBLE_DEVICES=0,1 python main_lincls.py -a resnet50 --batch-size 256 --epochs 100  --dist-url 'tcp://localhost:10001'--multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --pretrained checkpoint_0199.pth.tar  /home/user/zhaowenyi/ImageNet100/
