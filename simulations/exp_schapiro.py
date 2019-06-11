import numpy as np
from models import SEM, clear_sem
from sklearn import metrics
import pandas as pd
from scipy.special import logsumexp

def logsumexp_mean(x):
    return logsumexp(x) - np.log(len(x))

def batch_experiment(sem_kwargs, n_train=1400, n_test=600, progress_bar=True):

    # define the graph structure for the experiment

    g = np.array([
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    ], dtype=float)

    # define the random vectors
    d = 25
    items = np.random.randn(15, d) / np.sqrt(d)

    # draw random walks on the graph
    def sample_pmf(pmf):
        return np.sum(np.cumsum(pmf) < np.random.uniform(0, 1))

    train_nodes = [np.random.randint(15)]
    for _ in range(n_train-1):
        train_nodes.append(sample_pmf(g[train_nodes[-1]] / g[train_nodes[-1]].sum()))
        
    # draw hamiltonian paths from the graph

    # this graph defines the same thing but a preference order as well
    # higher number are  c
    preferred_nodes = np.array([
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    ], dtype=float)

    def sample_hamilton(node0):
        is_visited = np.zeros(15, dtype=bool)
        counter = 0
        nodes = []
        while counter < (len(is_visited)):
            p = g[node0] * ~is_visited * preferred_nodes
            if np.sum(p) == 0:
                p = g[node0] * ~is_visited

            node0 = sample_pmf(p / np.sum(p))
            nodes.append(node0)
            is_visited[node0] = True
            counter += 1
        return nodes

    test_nodes = []
    node0 = np.random.randint(15)
    for _ in range(n_test / 15):
        test_nodes += sample_hamilton(node0)
        node0 = test_nodes[-1]

    # embed the vectors
    all_nodes = train_nodes + test_nodes
    x = []
    for node in all_nodes:
        x.append(items[node])
    x = np.array(x)
    
    sem_model = SEM(**sem_kwargs)
    sem_model.run(x, progress_bar=progress_bar)

    
    # prepared diagnostic measures
    clusters = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    node_cluster = []
    for node in test_nodes:
        node_cluster.append(clusters[node])
    node_cluster = np.array(node_cluster)
    
    all_node_cluster = []
    for node in all_nodes:
        all_node_cluster.append(clusters[node])
    all_node_cluster = np.array(all_node_cluster)
    all_boundaries_true = np.concatenate([[False], (all_node_cluster[1:] != all_node_cluster[:-1])])
    
    test_boundaries = sem_model.results.e_hat[n_train-1:-1] != sem_model.results.e_hat[n_train:]
    boundaries = sem_model.results.e_hat[:n_train-1] != sem_model.results.e_hat[1:n_train]
    
    test_bound_prob = sem_model.results.log_boundary_probability[n_train:]
    bound_prob = sem_model.results.log_boundary_probability[1:n_train]
    
    # pull the prediction error (Bayesian Suprise)
    
    test_pe = sem_model.results.surprise[n_train:]
    bound_pe = sem_model.results.surprise[1:n_train]
    
    # cache the correlation between log boundary probability and log surprise
    r = np.corrcoef(
        sem_model.results.log_boundary_probability, sem_model.results.surprise
    )[0][1]

    
    output =  {
        'Community Transitions (Hamilton)': np.exp(logsumexp_mean(test_bound_prob[all_boundaries_true[1400:]])),
        'Other Parse (Hamilton)': np.exp(logsumexp_mean(test_bound_prob[all_boundaries_true[1400:]==False])),
        'Community Transitions (All Other Trials)': np.exp(logsumexp_mean(bound_prob[all_boundaries_true[1:n_train]])),
        'Other Parse (All Other Trials)': np.exp(logsumexp_mean(bound_prob[all_boundaries_true[1:n_train]==False])),
        'PE Community Transitions (Hamilton)': logsumexp_mean(test_pe[all_boundaries_true[1400:]]),
        'PE Other Parse (Hamilton)': logsumexp_mean(test_pe[all_boundaries_true[1400:]==False]),
        'PE Community Transitions (All Other Trials)': logsumexp_mean(bound_pe[all_boundaries_true[1:n_train]]),
        'PE Other Parse (All Other Trials)': logsumexp_mean(bound_pe[all_boundaries_true[1:n_train]==False]),
        'r':r
    }
    
    # clear_sem_model
    clear_sem(sem_model)
    sem_model = None

    return output
