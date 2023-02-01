# -*- coding: utf-8 -*-
# @Time    : 2021-12-11 13:48
# @Author  : Wily
# @File    : builder.py
# @Software: PyCharm



import torch
import torch.nn as nn
import diffdist
from .losses import SupCluLoss
import numpy as np

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, mlp_dim=128, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        self.criterion_clu = SupCluLoss(temperature=0.3)
        self.criterion_loc = nn.CrossEntropyLoss()
        # build encoders
        self.base_encoder = base_encoder(num_classes=256)
        self.momentum_encoder = base_encoder(num_classes=256)

        #self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        self.predictor = self._build_mlp(2, 256, 4096, 256, False)

        #dim_mlp = 2048
        #self.base_encoder.fc_clu = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))
        #self.momentum_encoder.fc_clu = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))

        #crop
        self.base_encoder.fc_crop = self._build_mlp(2, 2048, 4096, 256)
        self.momentum_encoder.fc_crop = self._build_mlp(2, 2048, 4096, 256)


        self.base_encoder.fc_loc = nn.Linear(2048, 2)
        self.momentum_encoder.fc_loc = nn.Linear(2048, 2)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass



    @torch.no_grad()
    def _batch_gather_ddp(self, images):
        """
        gather images from different gpus and shuffle between them
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        images_gather = []
        for i in range(2):
            #print(i, np.array(images).shape)
            #print(images[i].shape)
            batch_size_this = images[i].shape[0]
            images_gather.append(concat_all_gather(images[i]))
            batch_size_all = images_gather[i].shape[0]
        num_gpus = batch_size_all // batch_size_this

        n, c, h, w = images_gather[0].shape
        #print(n)#256
        permute = torch.randperm(n * 2).cuda()
        torch.distributed.broadcast(permute, src=0)
        images_gather = torch.cat(images_gather, dim=0)
        images_gather = images_gather[permute, :, :, :]
        # col1 = ([images_gather[0:n]])
        # col2 = ([images_gather[n:2*n]])
        col1 = torch.cat([images_gather[0:n//2], images_gather[n//2: n]], dim=3)
        col2 = torch.cat([images_gather[n:3* n//2], images_gather[3 * n//2:2 * n]], dim=3)
        images_gather = torch.cat([col1, col2], dim=2)

        bs = images_gather.shape[0] // num_gpus
        gpu_idx = torch.distributed.get_rank()

        return images_gather[bs * gpu_idx:bs * (gpu_idx + 1)], permute, n

    @torch.no_grad()
    def _batch_gather_ddp2(self, images, permute):
        """
        gather images from different gpus and shuffle between them
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        images_gather = []
        for i in range(2):
            #print(i)
            batch_size_this = images[i].shape[0]
            images_gather.append(concat_all_gather(images[i]))
            batch_size_all = images_gather[i].shape[0]
        num_gpus = batch_size_all // batch_size_this

        n, c, h, w = images_gather[0].shape
        torch.distributed.broadcast(permute, src=0)
        images_gather = torch.cat(images_gather, dim=0)
        images_gather = images_gather[permute, :, :, :]
        #col1 = ([images_gather[0:n]])
        #col2 = ([images_gather[n:2*n]])
        col1 = torch.cat([images_gather[0:n//2], images_gather[n//2: n]], dim=3)
        col2 = torch.cat([images_gather[n:3* n//2], images_gather[3 * n//2:2 * n]], dim=3)
        images_gather = torch.cat([col1, col2], dim=2)

        bs = images_gather.shape[0] // num_gpus
        gpu_idx = torch.distributed.get_rank()

        return images_gather[bs * gpu_idx:bs * (gpu_idx + 1)], permute, n

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k, labels):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        #N = logits.shape[0]  # batch size per GPU
        #labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)


    def forward_(self, q):
        q_gather = concat_all_gather_crop(q)
        n,c,h,w = q_gather.shape
        c1,c2 = q_gather.split([1,1],dim=2)
        f1,f2 = c1.split([1,1],dim=3)
        f3,f4 = c2.split([1,1],dim=3)
        # f1_ = (f1+f2)/2
        # f2_ = (f3+f4)/2
        q_gather = torch.cat([f1,f2,f3,f4],dim=0)
        q_gather = q_gather.view(n*4,-1)
        return q_gather

    def forward(self, x1, x2, m):
        images_gather1, permute1, bs_all1 = self._batch_gather_ddp(x1)
        images_gather2, permute2, bs_all2 = self._batch_gather_ddp2(x2, permute1)
        q1 = self.base_encoder(images_gather1)
        #print(q1.shape)  # torch.Size([64, 2048, 2, 2])
        q1_gather = self.forward_(q1)
        #print(q1_gather.shape) #torch.Size([256, 2048])


        q2 = self.base_encoder(images_gather2)
        q2_gather = self.forward_(q2)


        # dec branch
        label_loc = torch.LongTensor([0] * bs_all1 + [1] * bs_all1).cuda()
        label_loc = label_loc[permute1]
        q_loc = self.base_encoder.fc_loc(q1_gather)
        #print(q_loc.shape) #torch.Size([256, 2])
        loss_loc = self.criterion_loc(q_loc, label_loc)


        # con branch
        label_crop_ = permute1 % bs_all1
        label_crop = torch.cat([label_crop_, label_crop_], dim=0)
        cq1 = self.predictor(self.base_encoder.fc_crop(q1_gather)) #(cq1.shape) torch.Size([1024, 256])
        cq2 = self.predictor(self.base_encoder.fc_crop(q2_gather))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            ck1 = self.momentum_encoder.fc_crop(self.forward_(self.momentum_encoder(images_gather1)))
            ck2 = self.momentum_encoder.fc_crop(self.forward_(self.momentum_encoder(images_gather2)))

        logits_1 = torch.cat([cq1, ck2], dim=0)
        logits_1 = nn.functional.normalize(logits_1, dim=1)
        logits_2 = torch.cat([cq2, ck1], dim=0)
        logits_2 = nn.functional.normalize(logits_2, dim=1)

        return  loss_loc, 1 * (self.criterion_clu(logits_1, label_crop) + self.criterion_clu(logits_2, label_crop))



class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = 2048
        pass
        #del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        # self.base_encoder.fc_crop = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        # self.momentum_encoder.fc_crop = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        #self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def concat_all_gather_crop(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    tensors_gather = diffdist.functional.all_gather(tensors_gather, tensor, next_backprop=None, inplace=True)

    output = torch.cat(tensors_gather, dim=0)
    return output
