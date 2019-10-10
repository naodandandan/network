import tensorflow as tf
import numpy as np
import networkx as nx
import copy
import os
import csv
from sklearn.model_selection import KFold


def sim(a,b):
    set_a = set(a)
    set_b = set(b)
    sim = 1 - float(len( (set_a | set_b) - (set_a & set_b) ) )/ len(set_a | set_b)
    return sim


def cal_adj(self):
    adj = np.zeros((self.node_size, self.node_size), dtype=np.float16)
    with open("./data/all.csv", "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            adj[int(i[0])-1,int(i[1])-1] = 1
    return adj


def cal_sim(self):
    file_name = "./data/adj_mat_sim_gama"+ str(self.gama)+ "_label.txt"
    print(file_name)
    if os.path.isfile(file_name):
        print("loading " + file_name)
        adj_mat_sim = np.loadtxt(file_name, delimiter=',')
    else:
        print("calculating adj_mat_sim")
        np.set_printoptions(suppress=True)

        nodelist = list(self.g.G)
        nlen = len(nodelist)
        index = dict(zip(nodelist, range(nlen)))
        adj_mat_sim = copy.deepcopy(self.adj_mat)

        for i in range(0, self.node_size):
            for j in range(0, self.node_size):
                if j != i and i < self.drug_size and j < self.drug_size:
                    adj_mat_sim[index[str(i + 1)],index[str(j + 1)]] = self.adj_mat[index[str(i + 1)],index[str(j + 1)]] \
                                                                       + self.gama * sim(self.g.G[str(i + 1)], self.g.G[str(j + 1)])
        adj_mat_sim = np.array(adj_mat_sim)
		
        np.savetxt(file_name, adj_mat_sim , fmt='%4f', delimiter=',')
    return adj_mat_sim


def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation

def miss_edges(adj, removed_ratio):
    if removed_ratio == 0:
      return adj
    else:
      lower_trangile = np.tril(adj)
      folds = KFold(n_splits=10, shuffle=True, random_state=0)
      DDI_folds = folds.split(np.where(lower_trangile == 1)[0])
      count = 0
      del_adj = copy.deepcopy(lower_trangile)
      for train_index, test_index in DDI_folds:
          count = count + 1
          print(len(np.where(lower_trangile == 1)[0]))
          del_adj[np.where(lower_trangile == 1)[0][test_index], np.where(lower_trangile == 1)[1][test_index]] = 0
          train_inter = del_adj + np.transpose(del_adj)
          print(count)
          if count >= removed_ratio * 10:
              break
      return train_inter

class MDAE(object):
    def __init__(self, graph, encoder_layer_list, alpha, beta, gama, mu, drug_size, 
                  learning_rate, batch_size=100, max_iter=500, adj_mat=None):


        self.g = graph
        self.drug_size = drug_size
        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(encoder_layer_list)
        self.encoder_layer_num = len(encoder_layer_list)+1

        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.mu = mu
        self.bs = batch_size
        self.max_iter = max_iter
        self.lr = learning_rate

        self.sess = tf.Session()
        self.vectors = {}
        self.adj_mat = nx.to_numpy_array(self.g.G)

        self.adj_mat_sim = cal_sim(self)
        self.embeddings = self.get_train()
        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def get_train(self):
        adj_mat = self.adj_mat

        AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
        Adj = tf.placeholder(tf.float32, [None, None], name='adj_mat')
        B = tf.placeholder(tf.float32, [None, self.node_size], name='b_mat')

        fc = AdjBatch
        scope_name = 'encoder'
        layer_collector = []

        with tf.name_scope(scope_name):
            for i in range(1, self.encoder_layer_num):
                print("encoder" + str(i))
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        scope_name = 'decoder'
        with tf.name_scope(scope_name):
            for i in range(self.encoder_layer_num-2, 0, -1):
                print("decoder" + str(i))
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)
            fc = fc_op(fc,
                       name=scope_name+str(0),
                       n_out=self.encoder_layer_list[0],
                       layer_collector=layer_collector,)

        _embeddings_norm = tf.reduce_sum(tf.square(_embeddings), 1, keepdims=True)


        L_1st = tf.reduce_sum(
            Adj * (
                    _embeddings_norm - 2 * tf.matmul(
                        _embeddings, tf.transpose(_embeddings)
                    ) + tf.transpose(_embeddings_norm)
            )
        )


        L_2nd = tf.reduce_sum(tf.square((AdjBatch - fc) * B))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.mu * (tf.reduce_sum(tf.square(param[0]) + tf.abs(param[0])))

        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(L)

        init = tf.global_variables_initializer()
        self.sess.run(init)


        for step in range(self.max_iter):
            index = np.random.randint(self.node_size, size=self.bs) 
            adj_batch_train = adj_mat[index, :] 
            adj_batch_train_new = self.adj_mat_sim[index,:]
            adj_mat_train = adj_batch_train_new[:, index]

            b_mat_train = 1.*(adj_batch_train <= 1e-10) + self.beta * (adj_batch_train > 1e-10)
            self.sess.run(train_op, feed_dict={AdjBatch: adj_batch_train,
                                               Adj: adj_mat_train,
                                               B: b_mat_train})
            if step % 20 == 0:
                print("step %i: %s" % (step, self.sess.run([L, L_1st, L_2nd],
                                                           feed_dict={AdjBatch: adj_batch_train,
                                                                      Adj: adj_mat_train,
                                                                      B: b_mat_train})))

        return self.sess.run(_embeddings, feed_dict={AdjBatch: adj_mat})

    def save_embeddings(self, filename):
        print(self.bs)
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
		