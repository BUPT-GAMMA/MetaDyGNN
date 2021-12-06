import numpy as np
import pandas as pd
import torch


def preprocess(data_name):
    src_l, dst_l, ts_list, idx_list = [], [], [], []

    idx = 0
    with open(data_name) as f:
        for line in f:
            e = line.strip('\n').split(' ')
            src = int(e[0])
            dst = int(e[1])
            ts = float(e[2])

            src_l.append(src)
            dst_l.append(dst)
            ts_list.append(ts)
            idx_list.append(idx)
            idx += 1

    return pd.DataFrame({'src': src_l,
                         'dst': dst_l,
                         'ts': ts_list,
                         'idx': idx_list})


def reindex(df):
    upper_src = df.src.max() + 1
    print('upper_src', upper_src)

    new_df = df.copy()
    print(new_df.src.max())
    print(new_df.dst.max())

    new_df.src += 1
    new_df.dst += 1
    new_df.idx += 1

    print(new_df.src.min())
    print(new_df.src.max())
    print(new_df.dst.min())
    print(new_df.dst.max())
    print(new_df.idx.min())
    print(new_df.idx.max())

    return new_df


def run(data_name):
    PATH = '../data/{0}/{0}.txt'.format(data_name)
    OUT_DF = '../data/{0}/ml_{0}.csv'.format(data_name)
    OUT_FEAT = '../data/{0}/ml_{0}.npy'.format(data_name)
    OUT_NODE_FEAT = '../data/{0}/ml_{0}_node.npy'.format(data_name)

    df = preprocess(PATH)
    new_df = reindex(df)

    feat = torch.empty((new_df.idx.max() + 1, 172))
    feat = torch.nn.init.xavier_uniform_(feat, gain=1.414)

    max_idx = max(new_df.src.max(), new_df.dst.max())
    # rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    rand_feat = torch.empty((max_idx + 1, feat.shape[1]))
    rand_feat = torch.nn.init.xavier_uniform_(rand_feat)

    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    # print(new_df[:5])
    print('exit')

    return


run('dblp')


