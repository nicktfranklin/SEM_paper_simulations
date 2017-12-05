# a self-contained test of Online Simple RNN at event segmentation
# includes comparison against Simple RNN
#
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn import metrics

# hack to import model from parent directory
# TODO fix with proper modules
import sys
import os.path
sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import SEM, KerasLDS, LinearDynamicSystem, KerasMultiLayerNN
from models import KerasSimpleRNN, KerasGRU, KerasSimpleOnlineRNN 

# SEM parameters
K = 20  # maximum number of event types
lmda = 10  # stickyness parameter
alfa = 1.00  # concentration parameter
beta = 0.1 # transition noise
eta =  0.1  # learning rate


# define plotting function
import seaborn as sns

def plot_segmentation(post, y):
    cluster_id = np.argmax(post, axis=1)
    cc = sns.color_palette('Dark2', post.shape[1])
    
    #fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 2]))
    for clt in cluster_id:
        idx = np.nonzero(cluster_id == clt)[0]
        #axes[0].scatter(x_train[idx, 0], x_train[idx, 1], color=cc[clt], alpha=.5)
        
    #axes[1].plot(post)
    y_hat = np.argmax(post, axis=1)
    print "Adjusted Mutual Information:", metrics.adjusted_mutual_info_score(y, y_hat)
    print "Adjusted Rand Score:", metrics.adjusted_rand_score(y, y_hat)
    print 
    print np.argmax(post, axis=1)
    print '...vs. truth'
    print y


def build_alternating_moving_events(N, beta=0.01):
    pi = np.array([0.4, 0.6])
    bs = [[1, 1], [-1, -1], [1, -1], [-1, 1]]  # starting point!
    ws = [np.array([[1, 0], [.5, 2]]), np.array([[-1, .2], [-.2, 1]])]
    Sigma = np.eye(2) * beta
    x = np.zeros((N, 2), dtype=np.float32)
    
    x[0, :] = np.random.multivariate_normal(bs[0], Sigma)
    for n in range(1, N/4):
        _x = x[n-1, :] + np.array([0.05, 0.05])
        x[n, :] = np.random.multivariate_normal(_x, Sigma)
        
    x[N/4, :] = np.random.multivariate_normal(bs[1], Sigma)
    for n in range(N/4+1, N/2):
        _x = x[n-1, :] + np.array([-0.05, -0.05])
        x[n, :] = np.random.multivariate_normal(_x, Sigma)
            
    x[N/2, :] = np.random.multivariate_normal(bs[2], Sigma)
    for n in range(N/2+1, N/4*3):
        _x = x[n-1, :] + np.array([-0.05, -0.05])
        x[n, :] = np.random.multivariate_normal(_x, Sigma)
        
    x[N/4*3, :] = np.random.multivariate_normal(bs[3], Sigma)
    for n in range(N/4*3+1, N):
        _x = x[n-1, :] + np.array([0.05, 0.05])
        x[n, :] = np.random.multivariate_normal(_x, Sigma)      
        
    return x

x_train = build_alternating_moving_events(100)
y = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int), 
                    np.ones(25, dtype=int)*2, np.ones(25, dtype=int)*3])

#plt.plot(x_train[:, 0], x_train[:, 1])



# Online Simple RNN
#
print '\n\n\n\n  -------------------------------- SIMPLE ONLINE RNN  -------------------------------\n\n\n'

sem_kwargs5 = dict(lmda=lmda, alfa=alfa, beta=beta,
                  f_class=KerasSimpleOnlineRNN, f_opts=dict(t=5, n_epochs=10))

sem5 = SEM(**sem_kwargs5)
post = sem5.run(x_train, K=K)
plot_segmentation(post, y)

tf.Session().close()


# Simple RNN -- TAKES FOR FUCKING EVER
#
print '\n\n\n\n  -------------------------------- SIMPLE RNN  -------------------------------\n\n\n'

sem_kwargs4 = dict(lmda=lmda, alfa=alfa, beta=beta,
                  f_class=KerasSimpleRNN, f_opts=dict(t=5))

sem4 = SEM(**sem_kwargs4)
post = sem4.run(x_train, K=K)
plot_segmentation(post, y)

