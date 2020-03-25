import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax
from scipy.ndimage import distance_transform_edt as distance
from torch import einsum, Tensor
from torch.autograd import Variable
from utils import losses
'''
def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target
'''


w=np.array([1,5,20,10])
w=torch.from_numpy(w).float().cuda()

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target, num_classes):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(num_classes)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class SmoothL1(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(SmoothL1, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target, **kwargs):
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])        
        loss = self.criterion(output.type(torch.float32), target.type(torch.float32))
        return loss

class Weighted_CrossEntropyLoss2d(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=255, reduction='mean', **kwargs):
        super(Weighted_CrossEntropyLoss2d, self).__init__()       
        self.weight = weight 
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.num_classes=num_classes
        
    def forward(self, output, target, weight=None):
        if self.weight == None:
            if type(weight) == type(None):
                self.CE.weight = get_weights(target, num_classes=self.num_classes)
            else:
                self.CE.weight=weight

        loss = self.CE(output, target)
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean', **kwargs):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255, **kwargs):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None, **kwargs):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

class ICNetLoss(nn.Module):
    """Cross Entropy Loss for ICNet"""
    def __init__(self, weight=None, ignore_index=255, reduction='mean', **kwargs):
        super(ICNetLoss, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, outputs, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=outputs[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.CE(scale_pred, target)

        scale_pred = F.upsample(input=outputs[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.CE(scale_pred, target)

        scale_pred = F.upsample(input=outputs[2], size=(h, w), mode='bilinear', align_corners=True)
        loss3 = self.CE(scale_pred, target)

        scale_pred = F.upsample(input=outputs[3], size=(h, w), mode='bilinear', align_corners=True)
        loss4 = self.CE(scale_pred, target)

        return loss1 + 0.4 * loss2 + 0.4 * loss3 + 0.4 * loss4


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, num_classes, se_loss=True, se_weight=0.2, aux=True,
                 aux_weight=0.4, weight=None, ignore_index=255, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.num_classes = num_classes
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, outputs, target):
        pred1 = outputs[0]
        pred2 = outputs[2]
        se_pred = outputs[1]
        se_target = self._get_batch_label_vector(target, num_classes=self.num_classes).type_as(pred1)
        loss1 = super(EncNetLoss, self).forward(pred1, target)
        loss2 = super(EncNetLoss, self).forward(pred2, target)
        loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
        return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, num_classes):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, num_classes))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=num_classes, min=0,
                               max=num_classes - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect

class Ohem(nn.Module):
    def __init__(self, num_classes, criterion='Weighted_CrossEntropyLoss2d', ignore_index=255, thresh=0.7, min_kept=5, weight=None, **kwargs):
        super(Ohem, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.num_classes=num_classes
        self.criterion = getattr(losses, criterion)(ignore_index = ignore_index, weight=weight, **kwargs )

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        weight = get_weights(target, num_classes=self.num_classes)
        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_index)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_index
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target, weight=weight)

# adapted from https://github.com/PkuRainBow/OCNet/blob/master/utils/loss.py
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=5, weight=None, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)


    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_index)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_index
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class OHEMSegmentationLosses(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, num_classes, se_loss=True, se_weight=0.2,
                 aux=True, aux_weight=0.4, weight=None,
                 ignore_index=255, **kwargs):
        super(OHEMSegmentationLosses, self).__init__(ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.num_classes = num_classes
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, outputs, target):
        pred1 = outputs[0]
        pred2 = outputs[2]
        se_pred = outputs[1]
        se_target = self._get_batch_label_vector(target, num_classes=self.num_classes).type_as(pred1)
        loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
        loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
        loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
        return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, num_classes):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, num_classes))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=num_classes, min=0,
                               max=num_classes-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


#https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    B = seg.shape[0]
    C = seg.shape[1]
    res = np.zeros_like(seg)
    for b in range(B):
        for c in range(C):
            posmask = seg[b,c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                res[b,c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index=255, **kwargs):
        super(BoundaryLoss, self).__init__()
        self.ignore_index = ignore_index
       
    def forward(self, output, target):        
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        #(B,C,H,W)
        output = F.softmax(output, dim=1)
        #probability
        valid = [x for x in range(output.size()[1]) if x != self.ignore_index]

        dist_map = one_hot2dist(target.cpu().numpy())
        dc = torch.tensor(dist_map[:, valid, ...], dtype=torch.float32, device='cuda:0')
        pc = output[:, valid, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)
        loss = multipled.mean()

        return loss

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, **kwargs):        
        super(GeneralizedDiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target):   
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        #(B,C,H,W)
        output = F.softmax(output, dim=1)
        #probability
        valid = [x for x in range(output.size()[1]) if x != self.ignore_index]
        
        pc = output[:, valid, ...].type(torch.float32)
        tc = target[:, valid, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss

#https://arxiv.org/abs/1812.07032
#In the experiment, alpha starts from 1 and decreases to 0.01 gradually
class GDLwithBL(nn.Module):
    def __init__(self, alpha=0.9, ignore_index=255, **kwargs):        
        super(GDLwithBL, self).__init__()
        self.gdl = GeneralizedDiceLoss(ignore_index)
        self.bl = BoundaryLoss(ignore_index)
        self.alpha = alpha
        assert (alpha<=1 and alpha>=0)

    def forward(self, output, target):   
        loss1 = self.gdl(output, target)
        loss2 = self.bl(output, target)
        return self.alpha*loss1 + (1-self.alpha)*loss2

#https://arxiv.org/pdf/1808.05238.pdf?forcedefault=true
class GDLwithFL(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, Lambda=0.5, ignore_index=255, weight=[1,5,20,10], **kwargs):        
        super(GDLwithFL, self).__init__()
        self.ignore_index = ignore_index
        if weight != None:            
            self.weight = torch.tensor(weight, device='cuda:0').unsqueeze(dim=0)
        self.use_weight = (weight != None)
        self.alpha = alpha
        self.beta = beta
        self.Lambda = Lambda
        #weight:now size = (1,num_class)
        #weight[0,c] = number of label images that has class c

    def forward(self, output, target):   
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        #(B,C,H,W)
        output = F.softmax(output, dim=1)
        #probability
        valid = [x for x in range(output.size()[1]) if x != self.ignore_index]
        C_ = len(valid)
        B = target.size()[0]
        N = target.size()[2] * target.size()[3]

        pc = output[:, valid, ...].type(torch.float32)
        tc = target[:, valid, ...].type(torch.float32)
        #(B,C_,H,W)

        epsilon = 1e-10
        #dice terms
        TPc = einsum("bcwh,bcwh->bc", pc, tc)
        FNc = einsum("bcwh,bcwh->bc", 1-pc, tc)
        FPc = einsum("bcwh,bcwh->bc", pc, 1-tc)

        dlc = TPc / (TPc + self.alpha*FNc + self.beta*FPc + epsilon)

        #focal terms
        flc = einsum("bcwh,bcwh,bcwh->bc", tc, (1-pc)**2, torch.log(pc))

        if self.use_weight:
            #weights
            w = torch.ones((B, C_), device='cuda:0')
            for b in range(B):
                for c in range(C_):
                    w[b,c] = 1 if target[b,c,...].byte().any() else 0

            w = w/(self.weight[:,valid].type(torch.float32) + epsilon)
            dlc = einsum("bc,bc->bc", w, dlc)
            flc = einsum("bc,bc->bc", w, flc)

        dlb = einsum("bc->b", dlc)
        flb = einsum("bc->b", flc)
        lossb = C_ - dlb - self.Lambda*flb/N

        return lossb.mean()

#Debug
if __name__ == '__main__':
    
    gdl = GeneralizedDiceLoss()
    output = torch.rand((1,3,16,16))
    target = torch.zeros((1,16,16),dtype=torch.long)
    target[0,4:8,4:8] = 1
    target[0,8:12,8:12] = 2
    loss = gdl.forward(output, target)
    

    print('asd')