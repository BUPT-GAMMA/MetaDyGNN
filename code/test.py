# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35
import math
import os
import pickle
import torch

import numpy as np
import multiprocessing as mp
import random

# def dcg_at_k(scores):
#     # assert scores
#     return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))  # ind+1!!!
#
#
# def ndcg_at_k(real_scores, predicted_scores):
#     assert len(predicted_scores) == len(real_scores)
#     idcg = dcg_at_k(sorted(real_scores, reverse=True))
#     return (dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0
#
#
# def ranking(real_score, pred_score, k_list):
#     # ndcg@k
#     ndcg = {}
#     for k in k_list:
#         sorted_idx = sorted(np.argsort(real_score)[::-1][:k])
#         r_s_at_k = real_score[sorted_idx]
#         p_s_at_k = pred_score[sorted_idx]
#
#         ndcg[k] = ndcg_at_k(r_s_at_k, p_s_at_k)
#     return ndcg
#
#
# predicted1 = [.4, .1, .8]
# predicted2 = [.0, .1, .4]
# predicted3 = [.4, .1, .0]
# actual = [.8, .4, .1, .0]
#
# print(ranking(np.array(actual), np.array(predicted1), [1,3]))
# print(ranking(np.array(actual), np.array(predicted2), [1,3]))
# print(ranking(np.array(actual), np.array(predicted3), [1,3]))
#
# print(dcg_at_k([3,2,3,0,1,2]))
# print(ranking(np.array([3,3,2,2,1,0]), np.array([3,2,3,0,1,2]), [6]))


# def job(x):
#     return x*x, x+x
#
#
# def multicore():
#     l = []
#
#     pool = mp.Pool()
#     res = pool.map(job, range(10))
#     for r in res:
#         l.append(r[0])
#     print(res)
#     print(l)
#
#
# if __name__ == '__main__':
#     # multicore()
#     data_dir = os.path.join('../data', 'yelp')
#     supp_xs =pickle.load(open("{}/{}/support_ubtb_0.pkl".format(data_dir, 'meta_training')))
#     print(supp_xs)


from torch import tensor as ts
import numpy as np


# a = np.array([0.2, 0.5, 0.4, 0.3, 0.6, 0.1])
# b = np.array([0, 1, 1, 0, 1, 0])
# c = list()
# for i in a:
#     if i in sorted(a)[-b.sum():]:
#         c.append(1)
#     else:
#         c.append(0)
#
# print(sorted(a))
# print(sorted(a)[-b.sum():])
# print(c)

#
# from DataHelper import DataHelper
# #
# datahelper = DataHelper('reddit', k_shots=10, all_query=False)
#
# x, y, z, full_ngh_finder, train_ngh_finder = datahelper.load_data()
#
# print(type(y))
# print(y[0])
# print(len(y))
#
# support_x, support_y, query_x, query_y = zip(*y)
#
# print(type(support_x))
# print(len(support_x))
#
# print('support_x[0]', support_x[0])
#
# src_idx_l = np.array(support_x).astype(int).transpose()[0]
# target_idx_l = np.array(support_x).astype(int).transpose()[1]
# cut_time_l = np.array(support_x).transpose()[2]
#
# print(src_idx_l)
# print(type(src_idx_l))
# print(support_y[0])

# a = [torch.tensor(0.1), torch.tensor(0.2)]
#
# b = torch.stack(a)
#
# print(a, b)

# print(list(range(3)))

# car = {
#   "brand": "Porsche",
#   "model": "911",
#   "year": 1963
# }
#
# x = car.items()
#
# print(x)

# softmax = torch.nn.Softmax(dim=2)
#
# print(softmax(torch.tensor([0, 0, 0, 0, 0])))

a = 9
b = 8
c = 9
d = 7
e = 7

for i in range(30):
    _a = b + c + d
    _b = a + c
    _c = a + b + e
    _d = a + e
    _e = c + d

    a = _a
    b = _b
    c = _c
    d = _d
    e = _e
    print(a / 9, b / 8, c / 9, d / 7, e / 7)

