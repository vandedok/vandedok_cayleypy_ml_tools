import torch
from math import sqrt
from torch import nn

class AssymMSELoss(nn.Module):

    def __init__(self, tau=0.5, pred_reg_coef=None, pred_reg_target=None):
        super(AssymMSELoss, self).__init__()    
        self.left = 1 - tau
        self.right = tau
        assert (pred_reg_coef is None) == (pred_reg_target is None), "Both pred_reg_coef and pred_reg_target must be None or not None"
        self.pred_reg_coef = pred_reg_coef
        self.pred_reg_target = pred_reg_target

    def scale_loss_train(self, loss, y_norm, n):
        return sqrt(loss) * y_norm
    
    def scale_loss_val(self, loss, y_norm, n):
        return sqrt(loss)

    def forward(self, gt, pred, weights=None):
        diff = gt - pred
        squares = torch.pow(diff, 2)
        loss = torch.where(diff>0, squares*self.right, squares*self.left)
        if weights is not None:
            loss = loss * weights

        if self.pred_reg_coef:
            loss += self.pred_reg_coef * (pred.mean() - self.pred_reg_target)
        return torch.mean(loss)
    

class AssymMAELoss(nn.Module):

    def __init__(self, tau=0.5, pred_reg_coef=None, pred_reg_target=None):
        super(AssymMAELoss, self).__init__()    
        self.left = 1 - tau
        self.right = tau
        assert (pred_reg_coef is None) == (pred_reg_target is None), "Both pred_reg_coef and pred_reg_target must be None or not None"
        self.pred_reg_coef = pred_reg_coef
        self.pred_reg_target = pred_reg_target

    def scale_loss_train(self, loss, y_norm, n):
        return loss * y_norm
    
    def scale_loss_val(self, loss, y_norm, n):
        return loss

    def forward(self, gt, pred, weights=None):
        diff = gt - pred
        abs_v = torch.abs(diff)
        loss = torch.where(diff>0, abs_v*self.right, abs_v*self.left)
        if weights is not None:
            loss = loss * weights

        if self.pred_reg_coef:
            loss += self.pred_reg_coef * (pred.mean() - self.pred_reg_target)
        return torch.mean(loss)