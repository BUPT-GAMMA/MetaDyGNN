# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35

import numpy as np
import pandas as pd
import random


class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            off_set_l.append(len(n_idx_l))

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]

                    assert (len(ngh_idx) <= num_neighbors)
                    assert (len(ngh_ts) <= num_neighbors)
                    assert (len(ngh_eidx) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k - 1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1]  # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est,
                                                                                                 ngn_t_est,
                                                                                                 num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors)  # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records


class DataHelper:

    def __init__(self, dataset, k_shots=10, full_node=False, model_name='time_interval_update'):

        self.g_df = pd.read_csv('../data/{0}/ml_{0}.csv'.format(dataset))
        self.e_feat = np.load('../data/{0}/ml_{0}.npy'.format(dataset))
        self.n_feat = np.load('../data/{0}/ml_{0}_node.npy'.format(dataset))

        self.dataset = dataset

        if dataset == 'dblp':
            # dblp [0.8846149999999999, 0.923077, 0.9615379999999999, 1.0]
            self.test_time = sorted(set(self.g_df.ts))[-2]
            self.val_time = sorted(set(self.g_df.ts))[-3]
        else:
            self.val_time, self.test_time = np.quantile(self.g_df.ts, [0.6, 0.8])

        self.full_node = full_node
        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e6)
        self.node_set = set()
        self.degrees = dict()
        self.k_shots = k_shots
        self.node2hist = dict()
        self.data_size = 0  # number of edges, undirected x2

        if dataset == 'dblp':
            src_l = self.g_df.src.values
            dst_l = self.g_df.dst.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values
        else:
            src_l = self.g_df.u.values
            dst_l = self.g_df.i.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values

        print('max node: ', max(max(dst_l), max(src_l)))

        random.seed(2021)

        self.node_set = set(src_l).union(set(dst_l))

        self.mask_node_set = (set(src_l[ts_l > self.val_time]).union(set(dst_l[ts_l > self.val_time]))
                              - set(src_l[ts_l <= self.val_time]).union(set(dst_l[ts_l <= self.val_time])))

        self.test_mask_node_set = (set(src_l[ts_l > self.test_time]).union(set(dst_l[ts_l > self.test_time]))
                                   - set(src_l[ts_l <= self.test_time]).union(set(dst_l[ts_l <= self.test_time])))

        self.valid_mask_node_set = self.mask_node_set - self.test_mask_node_set

        # print('self.mask_node_set', len(self.mask_node_set))
        # print('self.test_mask_node_set', len(self.test_mask_node_set))
        # print('self.valid_mask_node_set', len(self.valid_mask_node_set))

        if dataset == 'dblp':
            mask_src_flag = self.g_df.src.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
            mask_dst_flag = self.g_df.dst.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
        else:
            mask_src_flag = self.g_df.u.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
            mask_dst_flag = self.g_df.i.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])

        # meta training set edges flags
        self.valid_train_edges_flag = (ts_l <= self.val_time)
        # meta validation set nodes
        self.valid_val_flag = np.array(
            [(a in self.valid_mask_node_set or b in self.valid_mask_node_set) for a, b in zip(src_l, dst_l)])
        # meta testing set nodes
        self.valid_test_flag = np.array(
            [(a in self.test_mask_node_set or b in self.test_mask_node_set) for a, b in zip(src_l, dst_l)])

        # meta training all edges (include support set and query set) for calculate the distribution of the node degree
        train_src_l = src_l[self.valid_train_edges_flag]
        train_dst_l = dst_l[self.valid_train_edges_flag]
        train_ts_l = ts_l[self.valid_train_edges_flag]

        for src, dst, ts in zip(train_src_l, train_dst_l, train_ts_l):

            if src not in self.degrees:
                self.degrees[src] = 0
            if dst not in self.degrees:
                self.degrees[dst] = 0

            if src not in self.node2hist:
                self.node2hist[src] = list()
            if dst not in self.node2hist:
                self.node2hist[dst] = list()

            self.degrees[src] += 1
            self.degrees[dst] += 1

            self.node2hist[src].append((dst, ts))
            self.node2hist[dst].append((src, ts))

        for node in self.mask_node_set:
            self.degrees[node] = 0

        #         print('len(self.degrees)', len(self.degrees))
        #         print('len(self.node_set)', len(self.node_set))
        #         print(self.degrees)
        #         print(sum(list(self.degrees.values())))

        self.node_dim = len(self.node_set)
        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def load_data(self):

        if self.dataset == 'dblp':
            src_l = self.g_df.src.values
            dst_l = self.g_df.dst.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values
        else:
            src_l = self.g_df.u.values
            dst_l = self.g_df.i.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values

        # meta training all edges (include support set and query set)
        train_src_l = src_l[self.valid_train_edges_flag]
        train_dst_l = dst_l[self.valid_train_edges_flag]
        train_ts_l = ts_l[self.valid_train_edges_flag]
        train_e_idx_l = e_idx_l[self.valid_train_edges_flag]

        # check the set partition is right
        train_node_set = set(train_src_l).union(train_dst_l)
        assert (len(self.node_set - self.mask_node_set) == len(train_node_set))

        # sample for validation set
        valid_src_l = src_l[self.valid_val_flag]
        valid_dst_l = dst_l[self.valid_val_flag]
        valid_ts_l = ts_l[self.valid_val_flag]
        valid_e_idx_l = e_idx_l[self.valid_val_flag]

        # meta testing all edges (include support set and query set)
        test_src_l = src_l[self.valid_test_flag]
        test_dst_l = dst_l[self.valid_test_flag]
        test_ts_l = ts_l[self.valid_test_flag]
        test_e_idx_l = e_idx_l[self.valid_test_flag]

        for src, dst, ts in zip(valid_src_l, valid_dst_l, valid_ts_l):

            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        for src, dst, ts in zip(test_src_l, test_dst_l, test_ts_l):

            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        # ↑ prepared for sample task

        # sample train task
        train_support_x, train_support_y, train_query_x, train_query_y = [], [], [], []

        for node in train_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

            if self.degrees[node] < 4:
                continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            if len(self.node2hist[node]) >= self.k_shots:

                pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(len(pos) // 2):
                    support_y.append(1)
                    query_y.append(1)

                neg = self.negative_sampling(int(self.k_shots // 2), pos[int(self.k_shots // 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                train_support_x.append(support_x)
                train_support_y.append(support_y)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[self.k_shots - 1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-int(self.k_shots / 2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                train_query_x.append(query_x)
                train_query_y.append(query_y)

        # train_support_x = np.array(train_support_x)
        # train_support_y = np.array(train_support_y)
        # train_query_x = np.array(train_query_x)
        # train_query_y = np.array(train_query_y)

        train_data = list(zip(train_support_x, train_support_y, train_query_x, train_query_y))

        # sample validation task
        valid_support_x, valid_support_y, valid_query_x, valid_query_y = [], [], [], []

        for node in self.valid_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

            if self.degrees[node] < 4:
                continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            if len(self.node2hist[node]) >= self.k_shots:

                pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(len(pos) // 2):
                    support_y.append(1)
                    query_y.append(1)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[int(self.k_shots / 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                valid_support_x.append(support_x)
                valid_support_y.append(support_y)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[self.k_shots - 1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-int(self.k_shots / 2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                valid_query_x.append(query_x)
                valid_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)

        valid_data = list(zip(valid_support_x, valid_support_y, valid_query_x, valid_query_y))

        # sample test task
        test_support_x, test_support_y, test_query_x, test_query_y = [], [], [], []

        for node in self.test_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

            if self.degrees[node] < 4:
                continue

            if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:

                pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(len(pos) // 2):
                    support_y.append(1)
                    query_y.append(1)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[int(self.k_shots / 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[self.k_shots - 1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-int(self.k_shots / 2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

            elif len(self.node2hist[node]) >= self.k_shots and self.full_node is True:

                pos = sorted(self.node2hist[node], key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(self.k_shots // 2):
                    support_y.append(1)
                for i in range(len(pos)-self.k_shots//2):
                    query_y.append(1)

                neg = self.negative_sampling(self.k_shots // 2, pos[int(self.k_shots / 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                neg = self.negative_sampling(len(pos)-self.k_shots//2, pos[-1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-(len(pos)-self.k_shots//2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)

        test_data = list(zip(test_support_x, test_support_y, test_query_x, test_query_y))

        max_idx = len(self.node_set)

        full_adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))

        full_ngh_finder = NeighborFinder(full_adj_list, uniform=True)

        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))

        train_ngh_finder = NeighborFinder(adj_list, uniform=False)

        print("train_set: ", len(train_data))
        print("valid_set: ", len(valid_data))
        print("test_set: ", len(test_data))

        return train_data, valid_data, test_data, full_ngh_finder, train_ngh_finder

    def load_data_in_time_sp(self, interval=4):

        if self.dataset == 'dblp':
            src_l = self.g_df.src.values
            dst_l = self.g_df.dst.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values
        else:
            src_l = self.g_df.u.values
            dst_l = self.g_df.i.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values

        # meta training all edges (include support set and query set)
        train_src_l = src_l[self.valid_train_edges_flag]
        train_dst_l = dst_l[self.valid_train_edges_flag]
        train_ts_l = ts_l[self.valid_train_edges_flag]
        train_e_idx_l = e_idx_l[self.valid_train_edges_flag]

        # check the set partition is right
        train_node_set = set(train_src_l).union(train_dst_l)
        assert (len(self.node_set - self.mask_node_set) == len(train_node_set))

        # sample for validation set
        valid_src_l = src_l[self.valid_val_flag]
        valid_dst_l = dst_l[self.valid_val_flag]
        valid_ts_l = ts_l[self.valid_val_flag]
        valid_e_idx_l = e_idx_l[self.valid_val_flag]

        # meta testing all edges (include support set and query set)
        test_src_l = src_l[self.valid_test_flag]
        test_dst_l = dst_l[self.valid_test_flag]
        test_ts_l = ts_l[self.valid_test_flag]
        test_e_idx_l = e_idx_l[self.valid_test_flag]

        for src, dst, ts in zip(valid_src_l, valid_dst_l, valid_ts_l):

            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        for src, dst, ts in zip(test_src_l, test_dst_l, test_ts_l):

            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        # ↑ prepared for sample task

        # sample train task
        train_support_x, train_support_y, train_query_x, train_query_y = [], [], [], []

        for node in train_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

#             if self.degrees[node] < 4:
#                 continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            if len(self.node2hist[node]) >= 2 * self.k_shots:

                pos = random.sample(self.node2hist[node], 2 * self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg

                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                train_support_x.append(support_x)
                train_support_y.append(support_y)

                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[self.k_shots:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                train_query_x.append(query_x)
                train_query_y.append(query_y)

        # train_support_x = np.array(train_support_x)
        # train_support_y = np.array(train_support_y)
        # train_query_x = np.array(train_query_x)
        # train_query_y = np.array(train_query_y)

        train_data = list(zip(train_support_x, train_support_y, train_query_x, train_query_y))

        # sample validation task
        valid_support_x, valid_support_y, valid_query_x, valid_query_y = [], [], [], []

        # print('mask_node_set:', len(mask_node_set))
        for node in self.valid_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

#             if self.degrees[node] < 4:
#                 continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            if len(self.node2hist[node]) >= 2 * self.k_shots:

                pos = random.sample(self.node2hist[node], 2 * self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg

                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                valid_support_x.append(support_x)
                valid_support_y.append(support_y)

                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[self.k_shots:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                valid_query_x.append(query_x)
                valid_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)

        valid_data = list(zip(valid_support_x, valid_support_y, valid_query_x, valid_query_y))

        # sample test task
        test_support_x, test_support_y, test_query_x, test_query_y = [], [], [], []

        # print('mask_node_set:', len(mask_node_set))
        for node in self.test_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

#             if self.degrees[node] < 4:
#                 continue

            if len(self.node2hist[node]) >= 2 * self.k_shots and self.full_node is False:

                pos = random.sample(self.node2hist[node], 2*self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg

                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[interval * pos_keys:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

            elif len(self.node2hist[node]) >= 2 * self.k_shots and self.full_node is True:

                # pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(self.node2hist[node], key=lambda x: x[1])  # from past(0) to now(1)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg

                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[self.k_shots:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)

        test_data = list(zip(test_support_x, test_support_y, test_query_x, test_query_y))

        max_idx = len(self.node_set)

        full_adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))

        full_ngh_finder = NeighborFinder(full_adj_list, uniform=True)

        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))

        train_ngh_finder = NeighborFinder(adj_list, uniform=False)

        print("train_set: ", len(train_data))
        print("valid_set: ", len(valid_data))
        print("test_set: ", len(test_data))

        return train_data, valid_data, test_data, full_ngh_finder, train_ngh_finder

    def get_node_dim(self):
        return self.node_dim

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 1
        for k in range(1, self.node_dim + 1):
            # print('k', k)
            # print('self.degrees[k]', self.degrees[k])
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        #         print('tot_sum', tot_sum)
        for k in range(self.neg_table_size):
            #             print('n_id', n_id)
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def negative_sampling(self, neg_size, cut_time):
        sampled_negs = []
        rand_idx = np.random.randint(0, self.neg_table_size, (neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        for i in sampled_nodes:
            sampled_negs.append((i, cut_time))
        return sampled_negs

