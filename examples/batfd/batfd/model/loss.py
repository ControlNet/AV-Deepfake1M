import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class MaskedBMLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :, :frame], true[i, :, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedFrameLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        # input: (B, T)
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :frame], true[i, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedContrastLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        # input: (B, C, T)
        loss = []
        for i, frame in enumerate(n_frames):
            # mean L2 distance squared
            d = torch.dist(pred1[i, :, :frame], pred2[i, :, :frame], 2)
            if labels[i]:
                # if is positive pair, minimize distance
                loss.append(d ** 2)
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip(self.margin - d, min=0.) ** 2)
        return torch.mean(torch.stack(loss))


class MaskedMSE(Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = MSELoss()

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :frame], true[i, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedBsnppLoss(Module):
    """Simplified version of BSN++ loss function."""

    def __init__(self, cbg_feature_weight=0.01, prb_weight_forward=1):
        super().__init__()
        self.cbg_feature_weight = cbg_feature_weight
        self.prb_weight_forward = prb_weight_forward

        self.cbg_loss_func = MaskedMSE()
        self.cbg_feature_loss = MaskedBMLoss(MSELoss())
        self.bsnpp_pem_reg_loss_func = self.cbg_feature_loss

    def forward(self, pred_bm_p, pred_bm_c, pred_bm_p_c, pred_start, pred_end,
        pred_start_backward, pred_end_backward, gt_iou_map, gt_start, gt_end, n_frames,
        feature_forward=None, feature_backward=None
    ):
        if self.cbg_feature_weight > 0:
            cbg_loss_forward = self.cbg_loss_func(pred_start, gt_start, n_frames) + \
                self.cbg_loss_func(pred_end, gt_end, n_frames)
            cbg_loss_backward = self.cbg_loss_func(torch.flip(pred_end_backward, dims=(1,)), gt_start, n_frames) + \
                self.cbg_loss_func(torch.flip(pred_start_backward, dims=(1,)), gt_end, n_frames)

            cbg_loss = cbg_loss_forward + cbg_loss_backward
            if feature_forward is not None and feature_backward is not None:
                inter_feature_loss = self.cbg_feature_weight * self.cbg_feature_loss(feature_forward,
                    torch.flip(feature_backward, dims=(2,)), n_frames)
                cbg_loss += inter_feature_loss
            else:
                inter_feature_loss = None
        else:
            cbg_loss = None
            cbg_loss_forward = None
            cbg_loss_backward = None
            inter_feature_loss = None

        prb_reg_loss_p = self.bsnpp_pem_reg_loss_func(pred_bm_p, gt_iou_map, n_frames)
        prb_reg_loss_c = self.bsnpp_pem_reg_loss_func(pred_bm_c, gt_iou_map, n_frames)
        prb_reg_loss_p_c = self.bsnpp_pem_reg_loss_func(pred_bm_p_c, gt_iou_map, n_frames)
        prb_loss = prb_reg_loss_p + prb_reg_loss_c + prb_reg_loss_p_c

        loss = cbg_loss + prb_loss if cbg_loss is not None else prb_loss
        return loss, cbg_loss, prb_loss, cbg_loss_forward, cbg_loss_backward, inter_feature_loss
