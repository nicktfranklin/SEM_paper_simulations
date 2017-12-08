import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from pandas import read_pickle
import pickle

# test datasets
from toy_data import Two2DGaussians
from toy_data import TwoAlternating2DGaussians
from toy_data import TwoLinearDynamicalSystems
from motion_data import MotionCaptureData
from coffee_shop_world_data import CoffeeShopWorldData

# hack to import model from parent directory
# TODO fix with proper modules
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import SEM, KerasLDS, LinearDynamicSystem, KerasMultiLayerPerceptron
from models import KerasSimpleRNN, KerasGRU
from opt.utils import evaluate
# end of imports -- this line is a total hack so we can get the imports for the jupyter notebooks. Pls don't remove

# lists of tests to run as (class name, params) pairs
#
tests_to_run = [
    ('Two2DGaussians', [4, 0.01]),
#    ('TwoAlternating2DGaussians', [100, 0.01]),
#    ('TwoLinearDynamicalSystems', [100, 0.01]),
#    ('MotionCaptureData', [10]),
    ('CoffeeShopWorldData', [])
]

# generate random string
#
def randstr(N=10):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

# export dictionary to file
#
def save_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

SUFFIX = randstr(10) # random token for output filenames
notebook_filename = 'test_notebook_' + SUFFIX + '.ipynb'

def get_results_filename(test_idx):
    return 'test_results_' + SUFFIX + '_' + str(test_idx) + '.pkl'

# get source code of given function
# also reformat it a bit
# func = function handle (whose code to take)
# self = what to replace all instances of self with
#
def get_code(func, self='self'):
    from inspect import getsourcelines
    code = getsourcelines(func)[0] # get source
    code = code[1:] # strip f'n title
    leading_spaces = len(code[0]) - len(code[0].lstrip())
    code = [x[leading_spaces:] for x in code] # remove indents 
    code = [x for x in code if 'return ' not in x] # remove return statement
    code = [x.replace('self.', self + '.') for x in code] # replace self. with something else
    return code

"""
Get the SEM hyperparameters based on the scene dimension D
"""
def get_Omega(D):
    t_horizon = 2 # time horizon to consider
    
    # parameters for schocastic gradient descent 
    sgd_kwargs = {
        'nesterov': True, 
        'lr': 0.1, 
        'momentum': 0.5, 
        'decay': 0.001
    }
    
    # specify the model architecture (makes a big difference! especially the training parameters)
    f_class = KerasSimpleRNN
    f_opts = dict(t=t_horizon, n_epochs=200, sgd_kwargs=sgd_kwargs,
                  n_hidden1=D, n_hidden2=D, 
                  hidden_act1='relu', hidden_act2='relu',
                  l2_regularization=0.01,
                  dropout=0.50)
    
    lmda = 10.
    alfa = 10.0
    
    # note! the likelihood function needs to be scaled with the dimensionality of the vectors
    # to compensate for the natural sharpening of the likelihood function as the dimensionality expands
    beta = 0.15 * D * np.var(test_data.X.flatten()) # this equals 1 if the data are scaled
    Omega = {
        'lmda': lmda,  # Stickyness (prior)
        'alfa': alfa, # Concentration parameter (prior)
        'beta': beta, # Likelihood noise
        'f_class': f_class,
        'f_opts': f_opts
    }
    print 'lambda =', lmda
    print 'alpha =', alfa
    print 'beta =', beta

    return Omega

"""
Run a single test with given TestData object
"""
def run_test(test_data, test_idx):
    # set the parameters for the models
    # TODO because we use D to define the parameters, we cannot fully decouple the parameters from the data set. So we have a chicken or egg problem here. Ideally, the user should be able to pass Omega and test_data here and they should be independent. For now, they're not
    Omega = get_Omega(test_data.D)
  
    K = test_data.X.shape[0]
    sem = SEM(**Omega)

    post, pe, log_lik, log_prior = sem.run(test_data.X, K=K, return_pe=True, return_lik_prior=True)
    
    mi, r = test_data.performance(post)

    results_filename = get_results_filename(test_idx)
    print 'Saving results to file ', results_filename
    res = {'test_data': test_data,
           'Omega': Omega,
           'sem': sem,
           'post': post,
           'pe': pe,
           'log_lik': log_lik,
           'log_prior': log_prior}
    save_pickle(res, results_filename)
    return mi, r


"""
Generate jupyter notebook for given tests
WARNING: there is a lot of coupling between this code and other test functions here and elsewhere.
"""
def write(tests_to_run, test_datas):
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    from nbformat import writes

    nb_cells = []

    # imports 
    # get them from this file
    code = []
    with open(__file__) as f:
        for line in f.readlines():
            if line.startswith('sys.'):
                # hack on top of a hack: __file__ doesn't work in jupyter
                code.append('sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)))\n') 
                continue
            if line.startswith('# end of imports'):
                break
            code.append(line)
    nb_cells.append(new_code_cell(source=code))

    # tests
    assert len(tests_to_run) == len(test_datas)
    for test_idx in range(len(tests_to_run)):
        test = tests_to_run[test_idx]
        test_data = test_datas[test_idx]

        test_name = test[0]
        test_params = test[1]
        test_params_str = [str(x) for x in test_params]

        # SEM initialization code
        #
        title = '# Running test ' + test_name + ' with params ' + ', '.join(test_params_str) + '\n'
        nb_cells.append(new_markdown_cell(source=title))

        code = ['# Load test data\n',
                '#\n',
                'np.random.seed(0) # for reproducibility\n',
                'test_data = ' + test_name + '(' + ','.join(test_params_str) + ')\n',
                'test_data.plot_scenes()\n']
        nb_cells.append(new_code_cell(source=code))

        code = get_code(get_Omega)
        code = ['# Set SEM parameters\n',
                '#\n',
                'D = test_data.D\n'] + code
        nb_cells.append(new_code_cell(source=code))

        # SEM run code
        #
        code = ['# Run SEM\n',
                '# \n',
                'K = test_data.X.shape[0]\n',
                'sem = SEM(**Omega)\n',
                '\n',
                'post, pe, log_lik, log_prior = sem.run(test_data.X, K=K, return_pe=True, return_lik_prior=True)\n']
        nb_cells.append(new_code_cell(source=code))

        # Alternative -- load everything from pickle file
        #
        results_filename = get_results_filename(test_idx)
        code = ["# Alternatively, load results from past execution\n",
                "#\n",
                "res = read_pickle('" + results_filename + "')\n",
                "sem = res['sem']\n",
                "post = res['post']\n"]
        nb_cells.append(new_code_cell(source=code))
    
        # performance code
        #
        code = get_code(test_data.performance, 'test_data')
        code = ['# Evaluate performance\n',
                '# \n'] + code
        nb_cells.append(new_code_cell(source=code))

        # plotting code
        #
        code = get_code(test_data.plot_segmentation, 'test_data')
        code = ['# Plot posterior\n',
                '# \n'] + code
        nb_cells.append(new_code_cell(source=code))

        code = get_code(test_data.plot_max_cluster, 'test_data')
        code = ['# Plot clusters\n',
                '# \n'] + code
        nb_cells.append(new_code_cell(source=code))

    # create notebook
    nb = new_notebook(cells=nb_cells)
    filestr = writes(nb, version=4)

    # write notebook to file
    print 'Saving as jupyter notebook to ', notebook_filename
    with open(notebook_filename, 'w') as f:
        f.write(filestr)


if __name__ == "__main__":
    tests = []

    # run tests
    for test in tests_to_run:
        test_name = test[0]
        test_params = test[1]
        np.random.seed(0) # for reproducibility
        test_data = getattr(sys.modules[__name__], test_name)(*test_params)
        tests.append(test_data) # must do it before running them... otherwise it doesn't work

    for test_idx in range(len(tests)):
        test_data = tests[test_idx]
        print '\n\n      Running ', test_data, '\n\n'
        mi, r = run_test(test_data, test_idx)

    # save jupyter notebook
    write(tests_to_run, tests)

