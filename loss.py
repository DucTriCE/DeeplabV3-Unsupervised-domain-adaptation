import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Optional, List
from functools import partial
from const import *


class TotalLoss(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, alpha1, gamma1, alpha2, gamma2, alpha3, gamma3):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''
        super().__init__()

        self.seg_tver_da = TverskyLoss(mode="multiclass", alpha=alpha1, beta=1-alpha1, gamma=gamma1, from_logits=True, teacher_update=False)
        self.seg_focal = FocalLossSeg(mode="multiclass", alpha=alpha3, gamma=gamma3, teacher_update=False)

        self.seg_tver_da_unsup = TverskyLoss(mode="multiclass", alpha=alpha1, beta=1-alpha1, gamma=gamma1, teacher_update=True)
        self.seg_focal_unsup = FocalLossSeg(mode="multiclass", alpha=alpha3, gamma=gamma3, teacher_update=True)

    def forward(self, outputs, targets, uncertainty_rate, teacher_update):

        seg_da = targets                    #[4,3,500,500]
        out_da = outputs                    #[4,3,500,500]
        
        _,seg_da = torch.max(seg_da, 1)     #[4,500,500]
        seg_da = seg_da.cuda()

        if not teacher_update:
            tversky_da_loss = self.seg_tver_da(out_da, seg_da, uncertainty_rate)
            focal_da_loss = self.seg_focal(out_da, seg_da, uncertainty_rate)
        else:
            tversky_da_loss = self.seg_tver_da_unsup(out_da, seg_da, uncertainty_rate)
            focal_da_loss = self.seg_focal_unsup(out_da, seg_da, uncertainty_rate)

        tversky_loss,focal_loss=tversky_da_loss,focal_da_loss
        loss = focal_loss+tversky_loss

        return focal_loss.item(),tversky_loss.item(),loss

def calc_iou(a, b):

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
   

    return IoU




def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    # https://github.com/qubvel/segmentation_models.pytorch/issues/612
    # logpt = F.binary_cross_entropy(output, target, reduction="none")
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLossSeg(_Loss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = 0.25, #Default none
        gamma: Optional[float] = 2, #Default 2.0
        teacher_update: bool = False,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.teacher_update = teacher_update

        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, uncertainty_rate) -> torch.Tensor:

        if self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0
            # y_pred ([4, 3, 500, 500])     y_true([4, 500, 500])
            
            if not self.teacher_update:
                for cls in range(num_classes):
                    cls_y_true = (y_true == cls).long()
                    cls_y_pred = y_pred[:, cls, ...]

                    if self.ignore_index is not None:
                        cls_y_true = cls_y_true[not_ignored]
                        cls_y_pred = cls_y_pred[not_ignored]

                    loss += self.focal_loss_fn(cls_y_pred, cls_y_true)
            elif self.teacher_update:
                
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 3, 1, 2)  # N, C, H*W

                values, indices = torch.max(y_pred, 1)  # Get max values and their indices
                labels_new = torch.where(values > uncertainty_rate, indices, torch.full_like(indices, num_classes))

                y_true_one_hot = F.one_hot(labels_new, num_classes + 1)
                y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)  
                y_true_one_hot = y_true_one_hot[:, :num_classes, :]

                y_pred = y_pred*y_true_one_hot
                y_true = y_true*y_true_one_hot

                loss += self.focal_loss_fn(y_pred, y_true)

        return loss

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        teacher_update: bool = False,
        eps: float = 1e-7
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.teacher_update = teacher_update

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, uncertainty_rate) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)         #ypred([4, 3, 500, 500]) ytrue([4, 500, 500])
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)                #([4, 250000])
            y_pred = y_pred.view(bs, num_classes, -1)   #([4, 3, 250000])

            if not self.teacher_update:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W
            elif self.teacher_update:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

                values, indices = torch.max(y_pred, 1)  # Get max values and their indices
                labels_new = torch.where(values > uncertainty_rate, indices, torch.full_like(indices, num_classes))

                y_true_one_hot = F.one_hot(labels_new, num_classes + 1)
                y_true_one_hot = y_true_one_hot.permute(0, 2, 1)  # Adjust dimensions to [batch, channel, height*width]
                y_true_one_hot = y_true_one_hot[:, :num_classes, :] #dua ra 1 neu du doan class voi doan >0.5

                # mask = y_true_one_hot != 0       #([4,3,5,5])
                y_pred = y_pred * y_true_one_hot
                y_true = y_true * y_true_one_hot

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()   #[4, 3, 250000]
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)

    return tversky_score

class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Positives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        teacher_update: bool = False,
        uncertainty_rate: float = 0.35,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0
    ):

        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, teacher_update, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)

if '__main__' == __name__:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='hyperparameters path')

    args = parser.parse_args()

    import torch.backends.cudnn as cudnn
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("Device: cuda")
        args.onGPU = True
        cudnn.benchmark = True

    import yaml
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    output = torch.rand(4,3,500,500)
    target = torch.rand(4,3,500,500)
    criteria = TotalLoss(hyp['alpha1'], hyp['gamma1'], hyp['alpha2'], hyp['gamma2'], hyp['alpha3'], hyp['gamma3'])

    with torch.cuda.amp.autocast():
        _,_,sup_loss = criteria(output,target, teacher_update=True)
