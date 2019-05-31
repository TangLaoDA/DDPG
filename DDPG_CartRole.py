"""

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym

#####################  hyper parameters  ####################

MAX_EPISODES = 200000
MAX_EP_STEPS = 2000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'CartPole-v0'

explore = 0.1#探索值（前期探索值较大，后期较小）

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a,self.ap = self._build_a(self.S, scope='eval', trainable=True)
            a_,a_p = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_1 = self._build_c(self.S_, tf.zeros(shape=[32,1]), scope='target', trainable=False)
            q_2 = self._build_c(self.S_, tf.ones(shape=[32,1]), scope='target', trainable=False)
            q_=tf.maximum(q_1,q_2)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = -(q_-q)*((tf.squeeze(tf.one_hot(self.a,2),axis=1)),self.ap)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        #self.a为确定值（DPG）
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        #随机选择数组索引
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        #获取随机副本
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs,self.S_: bs_})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # print(transition.shape)
        # exit()
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(net, 100, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(l2, 2, activation=tf.nn.softmax, name='a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return tf.expand_dims(tf.argmax(a,axis=1),axis=1),a

    def _build_c(self, s, a, scope, trainable):
        # print(type(a))
        # exit()

        a=tf.cast(a,dtype=tf.float32)
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            n_l1 = 100
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]  #4
a_dim = 2  #2

# print(s_dim)
# print(a_dim)
# print(a_bound)
# exit()

ddpg = DDPG(1, s_dim)

var = 3  # control exploration
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    alive_time=0
    for j in range(MAX_EP_STEPS):
        alive_time+=1
        if RENDER:
            env.render()

        # Add exploration noise
        #采用DPG输出确定值
        a = ddpg.choose_action(s)[0]


        # a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        if np.random.rand() < explore:
            a = np.random.randint(0, 2)
        s_, r, done, info = env.step(a)
        if done:
            alive_time=0
            r=-3

        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            # var *= .9995    # decay the action randomness
            explore -= 0.0001
            if explore < 0.0001:
                explore = 0.0001
            ddpg.learn()
            print("alive_time:  ",alive_time)
        if done:
            s = env.reset()
        else:
            s = s_
        ep_reward += r
        # if j == MAX_EP_STEPS-1:
        #     print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
        #     if ep_reward > -300:RENDER = False
