"""
    Greedy search for local optimal hyperparameters
    for our tensornet. Does not guarantee globally optimal result
"""

from Net.tensor.tensornet import TensorNet
import tensorflow as tf
from Net.tensor.inputdata import AMTData
from options import AMTOptions
import itertools
import numpy as np
import random
class GenericNet(TensorNet):
    def __init__(self, params):
        self.learning_rate = .003
        self.momentum  = params['momentum']
        self.bias_init = params['bias_init']
        self.weight_init = params['weight_init']
        self.channels = 3

        self.x = tf.placeholder('float', shape=[None, 250, 250, self.channels])
        self.y_ = tf.placeholder("float", shape=[None, 4])

        #self.w_conv1 = self.weight_variable([filter_size, filter_size, self.channels, depth])
        #self.b_conv1 = self.bias_variable([depth])
    
        #self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        self.last_layer = self.x
        



    def add_conv(self, filter_size, depth):
        last_layer_depth = self.last_layer.get_shape().as_list()[-1]
        w_conv = self.weight_variable([filter_size, filter_size, last_layer_depth, depth])
        b_conv = self.bias_variable([depth])
        h_conv = tf.nn.relu(self.conv2d(self.last_layer, w_conv) + b_conv)
        self.last_layer = h_conv

    def add_fc(self, num_fc_nodes):
        num_nodes = abs(TensorNet.reduce_shape(self.last_layer.get_shape()))
        flattened = tf.reshape(self.last_layer, [-1, num_nodes])
        
        w_fc = self.weight_variable([num_nodes, num_fc_nodes])
        b_fc = self.bias_variable([num_fc_nodes])

        h_fc = tf.nn.relu(tf.matmul(flattened, w_fc) + b_fc)
        self.last_layer = h_fc



    def append_output(self):
        num_nodes = abs(TensorNet.reduce_shape(self.last_layer.get_shape()))
        flattened = tf.reshape(self.last_layer, [-1, num_nodes]) 

        self.w_fc_out = self.weight_variable([num_nodes, 4])
        self.b_fc_out = self.bias_variable([4])

        self.y_out = tf.tanh(tf.matmul(flattened, self.w_fc_out) + self.b_fc_out)
        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.train = self.train_step.minimize(self.loss)

    def weight_variable(self, shape):
        return TensorNet.weight_variable(self, shape, self.weight_init)

    def bias_variable(self, shape):
        return TensorNet.bias_variable(self, shape, self.bias_init)
        

class SimAnneal():
    def __init__(self, archs, init_params):
        """
            hyp_params: full permutatinos of all hyperparameters, including init_params
            init_params: permutations of only initialization parameters (momentum, weight_init, bias_init)
        """
        self.archs = archs
        self.init_params = init_params

    
    def generate_net(self, param):
        """
            Given dictionary of parameters, generate and return a net architecture
            example: params = {
                'learning_rate': .01, 'momentum': .9, 'depth': 5, 'filter_size': 11,
                'num_convs': 2, 'num_fc': 2, 'num_fc_nodes': 128
            }
        
        """
        net = GenericNet(param)
        for i in range(param['conv']):
            net.add_conv(param['filters'][i], param['channels'][i])
        for i in range(param['fc']):
            net.add_fc(param['fc_dim'][i])
        net.append_output()
        return net

    @staticmethod
    def hamming_distance(vec1, vec2):
        dist = 0
        for x, y in zip(vec1, vec2):
            try:
                if not all(x == y):
                    dist += 1
            except:
                if not x == y:
                    dist += 1
        return dist

    @staticmethod
    def euclidean_distance(vec1, vec2):
        """
            computes the Euclidean distance between two vectors
        """
        vec1, vec2 = np.array(vec1),np.array(vec2)
        return np.linalg.norm(vec1 - vec2)**2.0
    
    @staticmethod
    def _is_iterable(val):
        try:
            it = iter(val)
            return True
        except TypeError as te:
            return False

    @staticmethod
    def param2vec(param):
        """
            Return the parameter as a vector in numpy array
        """
        vec = []
        for key in sorted(param.keys()):
            if not (key == 'momentum' or key == 'weight_init' or key == 'bias_init'):
                val = param[key]
                if SimAnneal._is_iterable(val):
                    val = sum(val)
                vec.append(val)
        return np.array(vec)
    
    @staticmethod
    def param2init_vec(param):
        return np.array( [ param['momentum'], param['weight_init'], param['bias_init'] ] )

    def nearest_init_params(self, curr, k=4):
        """
            Returns the k nearest neighbors looking only
            at the initialization parameters
        """
        all_params = self.init_params
        print len(self.init_params)
        vec1 = self.param2init_vec(curr)
        def euc_dist(param):
            vec2 = self.param2init_vec(param)
            dist = self.hamming_distance(vec1, vec2)
            if abs(dist) < 1e-9:
                return 1000.0       # make sure not getting same parameters back
            return dist

        all_params_sorted = sorted(all_params, key=euc_dist)
        return all_params_sorted[1:k+1]

     
    def nearest_archs(self, curr, k=9):
        """
            Return the k nearest neightbors looking
            at the overall architectures
        """
        all_params = self.archs
        vec1 = self.param2vec(curr)
        def euc_dist(param):
            vec2 = self.param2vec(param)
            dist = self.hamming_distance(vec1, vec2)
            if abs(dist) < 1e-3:
                return 1000.0
            return dist
        all_params_sorted = sorted(all_params, key=euc_dist)
        return all_params_sorted[len(self.init_params):len(self.init_params) + k]


    def nearest_neighbors(self, curr):
        """
            Returns the four nearest neighbors that have the
            same architectures but different initializations and momentums
            Also returns nine overall nearest neighbors
        """
        print "hello world" 

    @staticmethod    
    def merge(x, y):
        z = x.copy()
        z.update(y)
        return z

def generate_arch_permutations(hp):
    """
        Returns a list of dictionaries where
        each dictionary contains a unique permutation of the
        given net architectures
    """
    architectures = []
    for r in itertools.product(hp['conv'], hp['fc']):
        params = {
            'conv': r[0],
            'fc': r[1]
        }
        for i in range(len(hp['channels']) - params['conv'] + 1):
            copy = params.copy()
            for j  in range(len(hp['filters']) - params['conv'] + 1):
                for k in range(len(hp['fc_dim']) - params['fc'] + 1):
                    copy['fc_dim'] = hp['fc_dim'][k: k + params['fc']]
                    copy['filters'] = hp['filters'][j: j + params['conv']]
                    copy['channels'] = hp['channels'][i: i + params['conv']]
            architectures.append(copy)
    
    print "Generated " + str(len(architectures)) + " archs"

    return architectures


def generate_init_params_permutations(hp):
    """
        Geneterates a list of dictionaries where each dict is 
        a key value of only momentum, weight_init, and bias_init
    """

    init_params = []
    for r in itertools.product(hp['momentum'], hp['weight_init'], hp['bias_init']):
        param = {
            'momentum': r[0],
            'weight_init': r[1],
            'bias_init': r[2]
        }
        init_params.append(param)
    return init_params

def loss(param):
    """
        fake loss function
    """
    loss = 0.0
    for key in param.keys():
        val = param[key]
        try:
            loss += abs(sum([ abs(v) for v in val ]))         
        except:
            loss += val 
    return loss 

def test_simanneal(sa):
    """
        This is a super hacky test
        but whatever
    """

    min_loss = 1000.0
    max_loss = 0.0
    for arch in sa.archs:
        for ip in sa.init_params:
            min_loss = min(min_loss, loss(sa.merge(arch, ip)))
            max_loss = max(max_loss, loss(sa.merge(arch, ip)))
    print "Min loss in testing set: " + str(min_loss)
    print "Max loss in testing set: " + str(max_loss)

    curr_arch = sa.archs[3]
    curr_ip = sa.init_params[10]
    curr_loss = loss(sa.merge(curr_arch, curr_ip))
    print "Starting: " + str(sa.merge(curr_arch, curr_ip)) 
    print "Starting loss: " + str(curr_loss)

    for i in range(100):
        nearest_archs = sa.nearest_archs(curr_arch)
        nearest_ips = sa.nearest_init_params(curr_ip)
        tmp_arch = random.choice(nearest_archs)
        tmp_ip = random.choice(nearest_ips)
        tmp_loss = loss(sa.merge(tmp_ip, tmp_loss)
        if tmp_loss < curr_loss:
            curr_arch, curr_ip, curr_loss = tmp_arch, tmp_ip, tmp_loss

if __name__ == '__main__':

    hp = {
        "momentum": sorted([.09, .5, .9]),
        "weight_init": sorted([.005, .05, .5]),
        "bias_init": sorted([0.01, 0.1, .5]),
        "conv": sorted([1, 2, 3, 4]),   # number of convolutional layers
        "fc": sorted([0, 1, 2, 3]),     # number of fc layers not including output
        # reverse order matters for these since they go biggest->smallest and depend on num of fc an conv
        "channels": sorted([2, 3, 4, 5, 6], reverse=True),
        "filters": sorted([3, 5, 7, 11], reverse=True),
        "fc_dim": sorted([16, 50, 128, 400, 700], reverse=True)
    }


    with tf.Graph().as_default():
        archs = generate_arch_permutations(hp)
        init_params = generate_init_params_permutations(hp)

        sa = SimAnneal(archs, init_params)

        example_arch = SimAnneal.merge(archs[3], init_params[10])
        print example_arch
        sa.generate_net(example_arch)     
        sess = tf.Session()
        with sess.as_default():
            init = tf.initialize_all_variables()
            sess.run(init)
        sess.close()
        
        test_simanneal(sa)

        #print "Testing arch: " + str(example_arch)
        #similar_init_params= sa.nearest_init_params(example_arch)
        #similar_archs = sa.nearest_archs(example_arch)
        #print "resulting similar archs: "
        #for _ in similar_init_params:
        #    print str(_)

        #for _ in similar_archs:
        #    print str(_)




