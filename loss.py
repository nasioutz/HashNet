import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def pairwise_loss(outputs1,label1, outputs2, label2, config_loss=None):

    if config_loss is None:
        sigmoid_param = 1.0
        l_threshold = 15.0
        class_num = 1.0
    else:
        sigmoid_param = config_loss["sigmoid_param"]
        l_threshold = config_loss["l_threshold"]
        class_num = config_loss["class_num"]

    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = (torch.log(1+exp_product) - similarity * dot_product)
    loss =   (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) +
              torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num\
            + torch.sum(torch.masked_select(exp_loss, Variable(mask_en)))\
            + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))


def cauchy_cross_entropy_loss(outputs1, label1, outputs2=None, label2=None, config_loss=None):

    device = outputs1.device

    if config_loss is None:
        gamma = 1.0
        q_lambda = 0
        normed = False
    else:
        gamma = config_loss['gamma']
        q_lambda = config_loss['q_lambda']
        normed = config_loss['normed']

    if outputs2 is None:
        outputs2 = outputs1
        label2 = label1

    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()

    if normed:
        ip_1 = torch.mm(outputs1,outputs2.t())

        def reduce_sharper(t):
            return torch.sum(t,1).view(t.shape[0],1)

        mod_1 = torch.sqrt(torch.mm(reduce_sharper(outputs1.pow(2)), (reduce_sharper(outputs2.pow(2)) + 0.000001).t()))

        dist = outputs1.shape[1] / 2.0 * (1.0 - ip_1 / mod_1 + 0.000001)

    else:
        r_u1 = torch.sum(outputs1*outputs1, 1).view(-1,1)
        r_u2 = torch.sum(outputs2*outputs2, 1).view(-1,1)

        dist = r_u1 - 2 * torch.mm(outputs1,outputs2.t()) + r_u2.t() + 0.001


    cauchy = gamma / (dist + gamma)

    s_t = (similarity - 0.5) * 2.0
    sum_1 = torch.sum(similarity)
    sum_all = torch.sum(torch.abs(s_t))

    balance_param = torch.abs(similarity - 1.0) + sum_all / sum_1 * similarity
    mask = torch.eye(outputs1.shape[0]).to(device) == 0

    cauchy_mask = torch.masked_select(cauchy, Variable(mask))
    s_mask = torch.masked_select(similarity, Variable(mask))
    balance_p_mask = torch.masked_select(balance_param, Variable(mask))

    all_loss = - s_mask * torch.log(cauchy_mask) - (1 - s_mask) * torch.log(1 - cauchy_mask)

    cos_loss = torch.mean(all_loss * balance_p_mask)

    q_loss_image = torch.mean(torch.abs(outputs1-1.0).pow(2))
    q_loss = q_lambda * q_loss_image

    return torch.sum(all_loss * balance_p_mask) / list((all_loss * balance_p_mask).shape)[0]
    #return torch.mean((- s_mask * torch.log(cauchy_mask) - (1 - s_mask) * torch.log(1 - cauchy_mask)) * balance_p_mask)
    #return torch.mean(all_loss * balance_p_mask) #+ q_lambda * q_loss_image
    #return cos_loss + q_loss