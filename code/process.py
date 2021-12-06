import json
import numpy as np
import pandas as pd
import torch


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        # print(s)
        for idx, line in enumerate(f):
            # print(idx)
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df):
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df = df.copy()
    print('new_df.u.max()', new_df.u.max())
    print('new_df.i.max()', new_df.i.max())

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    print('new_df.u.max()', new_df.u.max())
    print('new_df.i.max()', new_df.i.max())

    return new_df


def run(data_name):
    PATH = '../data/{0}/{0}.csv'.format(data_name)
    OUT_DF = '../data/{0}/ml_{0}.csv'.format(data_name)
    OUT_FEAT = '../data/{0}/ml_{0}.npy'.format(data_name)
    OUT_NODE_FEAT = '../data/{0}/ml_{0}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df)

    print('feat.shape', feat.shape)
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    print('empty.shape', empty.shape)
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())

    # rand_feat = np.zeros((max_idx + 1, feat.shape[1]))

    rand_feat = torch.empty((max_idx + 1, feat.shape[1]))
    rand_feat = torch.nn.init.uniform_(rand_feat)
    rand_feat = torch.nn.init.xavier_uniform_(rand_feat, gain=1.414)

    print('feat.shape', feat.shape)
    print('rand_feat.shape', rand_feat.shape)

    # new_feat = np.zeros(feat.shape)
    # print('new_feat.shape', new_feat.shape)

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    # np.save(OUT_FEAT, new_feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    print('exit')

    return


# run('wikipedia')

run('reddit')

