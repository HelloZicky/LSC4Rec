from sklearn import metrics
import heapq
import numpy as np
import math
import itertools
import torch


def calculate_user_ndcg_hr(n=5, *buffer):
    user_num = 0
    ndcg_ = 0
    hr_ = 0
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = sorted(actions, key=lambda x: x[1], reverse=True)
        top_items = np.array(actions)[:n, :]
        num_postive = int(sum(np.array(actions)[:, 2]))
        if not 0 < num_postive:
            continue
        dcg = 0
        idcg = 0
        for i, (user_id, score, label) in enumerate(top_items):
            if label == 1:
                dcg += math.log(2) / math.log(i + 2)
            if i < num_postive:
                idcg += math.log(2) / math.log(i + 2)

        ndcg_ += dcg / idcg
        hr_ += 1 if any(item[2] for item in top_items) else 0
        user_num += 1

    ndcg = ndcg_ / user_num
    hr = hr_ / user_num
    return ndcg, hr


def calculate_overall_logloss(*buffer):
    prob, y = buffer
    logloss = float(metrics.log_loss(np.array(y), prob))

    return logloss


def calculate_overall_auc(*buffer):
    prob, y = buffer
    fpr, tpr, thresholds = [], [], []
    auc = float(metrics.roc_auc_score(np.array(y), prob))

    return auc, fpr, tpr, thresholds


def calculate_user_auc(*buffer):
    user_num = 0
    auc_ = 0
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = list(actions)
        prob = np.array(actions)[:, 1]
        y = np.array(actions)[:, 2]
        if not 0 < np.sum(y) < len(y):
            continue
        auc_ += float(metrics.roc_auc_score(np.array(y), prob))
        user_num += 1
    auc = auc_ / user_num
    return auc, user_num


def calculate_user_prec_mrr(topk, *buffer):
    user_num = 0
    auc_ = 0
    rel = []
    prec = 0
    mrr = 0
    rs = []
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = list(actions)
        actions = sorted(actions, key=lambda x: x[1], reverse=True)
        # print(actions)
        prob = np.array(actions)[:, 1]
        y = np.array(actions)[:, 2]
        if not 0 < np.sum(y) < len(y):
            continue
        # for index, (prob_, y_) in enumerate(zip(prob, y)):
        #     print(prob_ == y_)
        #     rel.append(prob_ == y_)
        # print(y)
        rel = y[:topk]
        # print("---")
        # print(topk)
        # print(len(rel))
        rs.append(rel)
        # auc_ += float(metrics.roc_auc_score(np.array(y), prob))
        prec += precision_at_k(rel, topk)
        user_num += 1
    prec = prec / user_num
    mrr = mean_reciprocal_rank(rs)
    return prec, mrr


def calculate_user_auc_misrec(*buffer):
    user_num = 0
    auc_ = 0
    fig1_list = []
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = list(actions)
        prob = np.array(actions)[:, 1]
        prob_trigger = np.array(actions)[:, 2]
        y = np.array(actions)[:, 3]
        mis_rec = np.mean(np.array(actions)[:, 4])
        auc_item = float(metrics.roc_auc_score(np.array(y), prob))
        auc_item_trigger = float(metrics.roc_auc_score(np.array(y), prob_trigger))
        auc_ += auc_item
        fig1_list.append([auc_item, auc_item_trigger, mis_rec])
        user_num += 1
    auc = auc_ / user_num
    return auc, fig1_list


def calculate_overall_acc(*buffer):
    prob, y = buffer
    acc = metrics.accuracy_score(y, prob)

    return acc


def calculate_overall_roc(*buffer):
    prob, y = buffer
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), prob, pos_label=1)

    return fpr, tpr


def calculate_overall_recall(*buffer):
    prob, y = buffer
    recall = metrics.recall_score(y, prob, average="macro")
    return recall


def calculate_overall_precision(*buffer):
    prob, y = buffer
    precision = metrics.precision_score(y, prob, average="macro")
    return precision


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.0
    return np.mean(r[: z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


if __name__ == '__main__':
    records = [
        [3, 0.151, 0],
        [3, 0.152, 0],
        [3, 0.103, 0],
        [3, 0.174, 1],
        [3, 0.135, 0],
        [3, 0.126, 0],
        [1, 0.151, 1],
        [1, 0.152, 0],
        [1, 0.103, 0],
        [1, 0.174, 0],
        [1, 0.135, 0],
        [2, 0.151, 0],
        [2, 0.152, 0],
        [2, 0.103, 1],
        [2, 0.174, 0],
        [2, 0.135, 0],
        [2, 0.126, 0],
    ]
    # ndcg, hr = calculate_user_ndcg_hr(np.array(records)[:, 0], np.array(records)[:, 1], np.array(records)[:, 2])
    # ndcg, hr = calculate_user_ndcg_hr(np.hstack(np.array(records)[:, 0], np.array(records)[:, 1], np.array(records)[:, 2]))
    # ndcg, hr = calculate_user_ndcg_hr(torch.Tensor(np.array(records)[:, 0]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 1]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 2]).view(-1, 1))
    # user_id_list = torch.Tensor([3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,]).view(-1, 1)
    # score_list = torch.Tensor([0.15, 0.15, 0.10, 0.17, 0.13, 0.12, 0.15, 0.15, 0.10, 0.17, 0.13, 0.15, 0.15, 0.10, 0.17, 0.13, 0.12,]).view(-1, 1)
    # label_list = torch.Tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,]).view(-1, 1)
    # print(torch.Tensor(np.array(records)[:, 0]).view(-1, 1))
    # ndcg, hr = calculate_user_ndcg_hr(torch.cat(torch.Tensor(np.array(records)[:, 0]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 1]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 2]).view(-1, 1), dim=1))
    # ndcg, hr = calculate_user_ndcg_hr(user_id_list, score_list, label_list)
    ndcg, hr = calculate_user_ndcg_hr(5, records)
    prec, mrr = calculate_user_prec_mrr(2, records)
    prec, mrr = calculate_user_prec_mrr(3, records)
    prec, mrr = calculate_user_prec_mrr(5, records)
    print(ndcg, hr)
    print(prec, mrr)
