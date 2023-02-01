## Learning What and Where to Learn: A New Perspective on Self-supervised Learning

### Introduction
This is a PyTorch implementation of [W2SSL] for self-supervised Learning.

### Main Results

The following results are based on ImageNet-1k self-supervised pre-training, followed by ImageNet-1k supervised training for linear evaluation. **Epochs** and **BS** suggests the self-supervised training epochs and the batch size of the corresponding methods. In column Mode, the PT, CL, MT indicate traditional pretext task based method, contrastive learning based method, and multi-task based method. In column **Pub. and Year**, the content (*e.g.,* 0.5X) of our method is the corresponding amount of forward computation compared with others. 

| Method       | Pub\. and Year | Mode | Epochs | BS   | Accuracy\(%\) |
|--------------|----------------|------|--------|------|---------------|
| Supervised   |                | \-   | \-     | \-   | 77\.2         |
| Colorization | ECCV2016       | PT   | 200    | 256  | 39\.6         |
| Jigpuz       | ECCV2016       | PT   | \-     | 256  | 45\.7         |
| Rotation     | ICLR2018       | PT   | 200    | 256  | 48\.1         |
| MoCo         | CVPR2020       | CL   | 200    | 256  | 60\.6         |
| SimCLR       | ICML2020       | CL   | 200    | 256  | 61\.9         |
| SimCLR       | ICML2020       | CL   | 200    | 1024 | 65\.3         |
| SimCLR       | ICML2020       | CL   | 800    | 4096 | 68\.9         |
| MoCo\-v2     | Arxiv2020      | CL   | 200    | 256  | 67\.5         |
| PCL          | ICLR2021       | CL   | 200    | 256  | 61\.5         |
| PCL\-v2      | ICLR2021       | CL   | 200    | 256  | 67\.6         |
| BYOL         | NIPs2020       | CL   | 200    | 4096 | 70\.6         |
| SwAV         | NIPs2020       | CL   | 400    | 256  | 70\.1         |
| InstLoc      | CVPR2021       | CL   | 200    | 256  | 61\.7         |
| DenseCL      | CVPR2021       | CL   | 200    | 256  | 63\.6         |
| MaskCo       | ICCV2021       | CL   | 200    | 256  | 65\.1         |
| SCRL         | CVPR2021       | CL   | 1000   | 256  | 70\.3         |
| MoCo\-v3     | ICCV2021       | CL   | 200    | 256  | 68\.1         |
| MoCo\-v3     | ICCV2021       | CL   | 300    | 4096 | 72\.8         |
| SimSiam      | CVPR2021       | CL   | 200    | 256  | 70\.0         |
| BarlowTwins  | ICCV2021       | CL   | 200    | 256  | 65\.0         |
| BarlowTwins  | ICCV2021       | CL   | 300    | 256  | 70\.7         |
| InfoMin      | ICLR2021       | CL   | 200    | 256  | 70\.1         |
| RegionCL     | Arxiv2021      | CL   | 200    | 256  | 69\.4         |
| XMOCO        | TCSVT2022      | CL   | 200    | 256  | 65\.0         |
| BatchFormer  | CVPR2022       | CL   | 200    | 256  | 68\.4         |
| HCSC         | CVPR2022       | CL   | 200    | 256  | 69\.2         |
| CCrop        | CVPR2022       | CL   | 200    | 256  | 67\.8         |
| DUPR         | TPAMI2022      | CL   | 200    | 256  | 63\.6         |
| SAT          | TPAMI2022      | CL   | 200    | 1024 | 72\.8         |
| Sela         | ICLR2020       | MT   | 200    | 256  | 61\.5         |
| DeepCluster  | ICCV2017       | MT   | 200    | 256  | 48\.4         |
| JigClu       | CVPR2021       | MT   | 200    | 256  | 66\.4         |
| GLNet        | TCSVT2022      | MT   | 200    | 256  | 70\.5         |
| LEWEL        | CVPR2022       | MT   | 200    | 256  | 68\.4         |
| W^2SSL       | 0\.50X         | MT   | 200    | 256  | 70\.8         |
| W^2SSL       | 0\.50X         | MT   | 200    | 1024 | 72\.8         |



**Pre-trained models** and **configs** can be found at [README.md](https://github.com/WilyZhao8/W2SSL/blob/main/README.md).


### Usage: Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo v3](https://github.com/facebookresearch/moco-v3), this repo contains minimal modifications on the official PyTorch ImageNet code. 

The code has been tested with CUDA 11.3, PyTorch 1.9.0 and timm 0.4.9.

### Usage: Self-supervised Pre-Training

#### ResNet-50 with 1-node (4-GPU) training, batch 256
run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py --moco-m-cos --crop-min=.2 --dist-url 'tcp://localhost:10109' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --batch-size 512 --epochs 200 --loss_crop 1.0 [your imagenet-folder with train and val folders]
```

#### ResNet-50 with 1-node (8-GPU) training, batch 1024
run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_moco.py --moco-m-cos --crop-min=.2 --dist-url 'tcp://localhost:10109' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --batch-size 2048 --epochs 200 --loss_crop 1.0 [your imagenet-folder with train and val folders]
```

#### Notes:
1. The batch size specified by `-b` is the total batch size across all GPUs, and --batch-size divided by 2 is the number of synthetic samples input to the model.
2. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).
3. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting, and by default uses automatic *mixed-precision* for pre-training.

### Usage: Linear Classification

By default, we use momentum-SGD and a batch size of 256 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python main_lincls.py -a resnet50 --batch-size 256 --epochs 100  --dist-url 'tcp://localhost:10016'--multiprocessing-distributed --world-size 1 --rank 0 --print-freq 100 --pretrained checkpoint_0199.pth.tar [your imagenet-folder with train and val folders]
```

### Transfer Learning

See the instructions in the [transfer](https://github.com/facebookresearch/moco-v3/tree/main/transfer) dir.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation
```
@Article{zhao2023w2ssl,
  author  = {Wenyi Zhao and Weidong Zhang and Wenhe Jia and Xipeng Pan and Huihua Yang*},
  title   = {Learning What and Where to Learn: A New Perspective on Self-supervised Learning},
  journal = {Submission status in IJCAI2023},
  year    = {2023},
}
```
