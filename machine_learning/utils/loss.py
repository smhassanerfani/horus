#a
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):
    def __init__(self, class_num=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore_index=255, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = class_num
        self.size_average = size_average
        self.num_classes = class_num
        self.one_hot = one_hot
        self.ignore = ignore_index
        self.weights = weight


    def forward(self, input, target, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        target_onehot = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        if self.weights != None:
            probs = (self.weights * probs * target_onehot).sum(1)
        else:
            probs = (probs * target_onehot).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLossUncert(nn.Module):
    def __init__(self, class_num=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(FocalLossUncert, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = class_num
        self.size_average = size_average
        self.num_classes = class_num
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight


    def forward(self, input, target, uncet, eps=1e-5):

        EPS = 1e-7
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        uncet = uncet.permute(0, 2, 3, 1).contiguous().view(-1)  # B * H * W, C = P, C

        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]
            uncet = uncet[valid]

        target_onehot = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        if self.weights != None:
            probs = (self.weights * probs * target_onehot).sum(1)
        else:
            probs = (probs * target_onehot).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p / (uncet + EPS)+ (uncet + EPS).log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes