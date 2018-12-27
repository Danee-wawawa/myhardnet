import torch
import torch.nn as nn
import sys
import numpy as np
from torch.autograd import Variable

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, anchor.size(0)) + torch.t(d2_sq.repeat(1, positive.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def distance_vectors_pairwise(anchor, positive, negative):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)
    n_sq = torch.sum(negative * negative, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
    d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
    return d_a_p, d_a_n, d_p_n

def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_L2Net(anchor, positive, anchor_swap = False,  margin = 1.0, loss_type = "triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    
    if loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos1);
        exp_den = torch.sum(torch.exp(2.0 - dist_matrix),1) + eps;
        loss = -torch.log( exp_pos / exp_den )
        if anchor_swap:
            exp_den1 = torch.sum(torch.exp(2.0 - dist_matrix),0) + eps;
            loss += -torch.log( exp_pos / exp_den1 )
    else: 
        print ('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet(anchor, positive, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, margin_neg = 0.5, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)

    test = torch.arange(0,pos1.size()[0])

    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        idx1 = torch.min(dist_without_min_on_diag,1)[1]

        min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
        idx2 = torch.min(dist_without_min_on_diag,0)[1]
        min_neg5 = torch.min(min_neg,min_neg2)        
        # print('----------------------------------------', idx1)
        # print('----------------------------------------', idx2)
        # mask1 = torch.eq(idx1,idx2)
        # print('---------------------------',mask,anchor)
        
        # for i in range(1024):
        #     test.append(i)
        # test = torch.cuda.FloatTensor(test)
        # print('----------------------------------------', test,mask1)
        # print('-------------------',mask1)
        # # mask2 = torch.cat((torch.t(mask1),torch.t(mask1)),1)
        # for i in range(1024):
        #     test.append(i)
        # test1 = torch.cuda.FloatTensor(np.array(test))
        
        # print('---------------',pos1,test.cuda(),mask1,idx1)
        # pos1_1 = torch.masked_select(pos1,mask1)
        # mask3 = torch.ByteTensor(mask1)

        mask1 = torch.eq(idx1,idx2)
        min_neg_1 = torch.masked_select(min_neg,mask1)
        min_neg2_1 = torch.masked_select(min_neg2,mask1)
        min_neg3 = torch.min(min_neg_1,min_neg2_1)
        pos1_1 = torch.masked_select(pos1,mask1)

        mask2 = torch.ne(idx1,idx2)
        min_neg_2 = torch.masked_select(min_neg,mask2)
        min_neg2_2 = torch.masked_select(min_neg2,mask2)
        min_neg4 = torch.min(min_neg_2,min_neg2_2)
        pos1_2 = torch.masked_select(pos1,mask2)
        


        idx1_1 = torch.masked_select(idx1,mask2)
        idx2_1 = torch.masked_select(idx2,mask2)
        anchor_a = torch.index_select(anchor,0,idx2_1)
        positive_p = torch.index_select(positive,0,idx1_1)
        # print('---------------',mask2.data)
        
        # mask3 = torch.masked_select(test.cuda(),mask2.data)
        # mask3 = Variable(mask3.type(torch.LongTensor).cuda())
        # print('---------------',mask2.cuda(),idx1)
        # anchor_a = torch.index_select(anchor,0,mask3)
        # positive_p = torch.index_select(positive,0,mask3)
        dist_matrix_test = distance_matrix_vector(positive_p, anchor_a) +eps
        bet_neg = torch.diag(dist_matrix_test)
        # pos1_1 = torch.masked_select(pos1,mask2)

        pos1_2_mean = torch.mean(pos1_2)
        min_neg4_mean = torch.mean(min_neg4)

        

        #pos1_2_mean = pos1_2_mean.data.cpu().numpy()
        #min_neg4_mean = min_neg4_mean.data.cpu().numpy()


        #print("-----------------",pos1_2_mean)
        pos12txt = open('../pos2.txt','a')
        #for j in pos1_2_mean:
        #    pos12txt.write(str(j))
        pos12txt.write(str(pos1_2_mean))
        pos12txt.write('\n')
        pos12txt.close()
        minneg4txt = open('../neg2.txt','a')
        #for v in min_neg4_mean:
        #    minneg4txt.write(str(v))
        minneg4txt.write(str(min_neg4_mean))
        minneg4txt.write('\n')
        minneg4txt.close()               


        
        # print('----------------------------------------', pos1_1)
        pos = pos1_2
        min_neg = min_neg4
        bet_neg = bet_neg
        margin_test = torch.mean(min_neg) - torch.mean(pos)
        # print('----------------------------------------', pos,min_neg,pos1_2,bet_neg)
        # min_neg = torch.min(min_neg,min_neg2)
        # positive_p = torch.index_select(positive,0,idx1)
        # anchor_a = torch.index_select(anchor,0,idx2)
        # dist_matrix_test = distance_matrix_vector(positive_p, anchor_a) +eps
        # bet_neg = torch.diag(dist_matrix_test)

        # min_neg = min_neg
        # pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        #loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #loss = torch.clamp(margin + pos - min_neg, min=0.0) + torch.clamp(margin - bet_neg, min=0.0)
        #loss1 = torch.clamp(margin_test + pos - min_neg, min=0.0)
        #loss2 = torch.clamp(0.5*margin_test + pos - bet_neg, min=0.0)
        loss1 = torch.clamp(1.0 + pos - min_neg, min=0.0)
        loss2 = torch.clamp(1.5 - min_neg, min=0.0)
        #loss2 = torch.clamp(margin_neg + pos1_2 - bet_neg, min=0.0)
        #print('-------------------------------------------------------------------------',margin_test)
        #print('--------------------------------------------------',loss2)
        #loss1 = torch.mean(loss1)
        #loss2 = torch.mean(loss2)
        loss = loss1+loss2
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    #loss = loss
    return loss


def global_orthogonal_regularization(anchor, negative):

    neg_dis = torch.sum(torch.mul(anchor,negative),1)
    dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis),2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0)
    
    return gor

