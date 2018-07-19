
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
#mport keras
#rom keras.utils import np_utils
# keras.datasets.mnist.load_data()
# (x_Train, y_Train), (x_Test, y_Test) = keras.datasets.mnist.load_data()

# plot_image(x_Train[9999])
# y_Train[9999]

# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# In[3]:


import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


# In[5]:


def sample_hidden_from_visible(vis_layer,w,bh,hrand):
    hprobs = tf.nn.sigmoid(tf.add(tf.matmul(vis_layer,w),bh))
    hstate = sample_prob(hprobs,hrand)
    return hprobs,hstate
def sample_prob(probs,rand):
    return tf.nn.relu((tf.sign(probs - rand)))
def sample_visible_from_hidden(hidden_layer,w,bv,vrand,visible_unit_type,n_feature,train_stddev):
    visible_activation = tf.add(tf.matmul(hidden_layer,tf.transpose(w)),bv)
    if visible_unit_type =='bin':
        vprobs = tf.nn.sigmoid(visible_activation)
    elif visible_unit_type =='guass':
        vprobs = tf.truncated_normal( (1,n_feature),mean=visible_activation,stddev=train_stddev)
    else:
        vprobs = None    
    return vprobs,sample_prob(vprobs,vrand)
def gibbs_sampling_step(visible,w,bh,hrand,bv,vrand,visible_unit_type,n_feature,train_stddev):
    hprobs,hstates = sample_hidden_from_visible(visible,w,bh,hrand)
    vprobs,_ = sample_visible_from_hidden(hprobs,w,bv,vrand,visible_unit_type,n_feature,train_stddev)
    hprobs1, hstates1 = sample_hidden_from_visible(vprobs,w,bh,hrand)
    return hprobs, hstates, vprobs, hprobs1, hstates1
def compute_positive_association(visible,hidden_states,hidden_probs):
    if visible_unit_type =='bin':
        positive = tf.matmul(tf.transpose(visible),hidden_states)
    elif visible_unit_type =='guass':
        positive = tf.matmul(tf.transpose(visible),hidden_probs)
    else:
        vprobs = None   
    return positive
def gen_mini_batches(X,batch_size):
    X = np.array(X)
    for i in range(0,X.shape[0],batch_size):
        yield X[i:i + batch_size]


# In[20]:


tf.reset_default_graph()    
num_hidden = 250
n = 784
num_epoch = 50
batch_size = 128
learning_rate = 0.0001
gibbs_sampling_steps = 3
train_w_stddev = 0.1
regcoef = 0.00001
visible_unit_type = 'bin'
x = tf.placeholder(dtype=tf.float64, shape=[None, n])
y = tf.placeholder(dtype=tf.float64, shape=[None, 10])
hrand  =  tf.placeholder(dtype=tf.float64, shape=[None, num_hidden])
vrand  =  tf.placeholder(dtype=tf.float64, shape=[None, n])
keep_prob = tf.placeholder(dtype=tf.float64)

w = tf.Variable(tf.truncated_normal([n, num_hidden], mean=0.0, stddev=train_w_stddev, dtype=tf.float64))
bh = tf.Variable(tf.constant(0.1,shape=[num_hidden],dtype=tf.float64))
bv = tf.Variable(tf.constant(0.1,shape=[n],dtype=tf.float64))

encode,_ = sample_hidden_from_visible(x,w,bh,hrand)
reconstruction,_ = sample_visible_from_hidden(encode,w,bv,vrand,visible_unit_type,n,train_w_stddev)
hprob0, hstate0, vprob, hprob1, hstate1 = gibbs_sampling_step(x,w,bh,hrand,bv,vrand,visible_unit_type,n,train_w_stddev)
positive = compute_positive_association(x,hprob0,hstate0)

nn_input = vprob
for step in range(gibbs_sampling_steps-1):
    hprob, hstate, vprob, hprob1, hstate1 = gibbs_sampling_step(nn_input,w,bh,hrand,bv,vrand,visible_unit_type,n,train_w_stddev)
    nn_input = vprob
negative = tf.matmul(tf.transpose(vprob),hprob1)    
w_upd8 = w.assign_add(learning_rate * (positive - negative) / batch_size)
bh_upd8 = bh.assign_add(learning_rate * tf.reduce_mean(tf.subtract(hprob0,hprob1),0))
bv_upd8 = bv.assign_add(learning_rate * tf.reduce_mean(tf.subtract(x     ,vprob ),0))
clip_inf = tf.clip_by_value(vprob,1e-10,float('inf'))
clip_sup = tf.clip_by_value(1 - vprob,1e-10,float('inf'))
#cross entropy
loss = - tf.reduce_mean(x*tf.log(clip_inf) + (1.0 - x)*tf.log(clip_sup))
cost = loss + regcoef*(tf.nn.l2_loss(w) + tf.nn.l2_loss(bh) + tf.nn.l2_loss(bv))


# In[21]:


x_Train = mnist.train.images
x_Test = mnist.test.images


# In[14]:


x_Train[9999]


# In[52]:


init = tf.global_variables_initializer()
ckpt_file = 'D:/pywork/RBM/'
saver = tf.train.Saver()
isTrain = False
with tf.Session() as sess:
    if isTrain:
        sess.run(init)
        for epoch in range(num_epoch):
            np.random.shuffle(x_Train)
            batches = [_ for _ in gen_mini_batches(x_Train,batch_size)]
            total_batches = len(batches)
            for idx, batch in enumerate(batches):
                cost_val, _, _, _ = sess.run([cost,w_upd8,bh_upd8,bv_upd8],feed_dict={
                    x:batch,hrand:np.random.rand(batch.shape[0],num_hidden),vrand:np.random.rand(batch.shape[0],batch.shape[1])
                })
            if (epoch*total_batches + idx) % 100 ==0:
                saver.save(sess,ckpt_file+'model.ckpt')
            print('{0}_{1}:cost={2}'.format(epoch, idx, cost_val))
    else:
        raw = x_Train[9999]
        batch = np.reshape(raw,[1,n])
        ckpt = tf.train.get_checkpoint_state(ckpt_file)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        vprob_out = sess.run(vprob,feed_dict={
            x:batch,hrand:np.random.rand(batch.shape[0],num_hidden),vrand:np.random.rand(batch.shape[0],batch.shape[1])
        })
        w1 = sess.run(w)
        w1t = w1.T
        print(len(w1t))

        fig = plt.gcf()
        fig.set_size_inches(12, 14)
        for i in range(0, 25):
            ax=plt.subplot(5,5, 1+i)
            imgw = np.reshape(w1t[i],[28,28])
            ax.imshow(imgw, cmap='binary')
            title= "label"
            ax.set_title(title,fontsize=10) 
            ax.set_xticks([]);ax.set_yticks([])        
      
        plt.show()

