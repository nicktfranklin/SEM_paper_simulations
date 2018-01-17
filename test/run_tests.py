import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from pandas import read_pickle
import pickle
from keras.models import model_from_json

# test datasets
from toy_data import Two2DGaussians
from toy_data import TwoAlternating2DGaussians
from toy_data import TwoLinearDynamicalSystems
from motion_data import MotionCaptureData
from coffee_shop_world_data import CoffeeShopWorldData

# hack to import model from parent directory
# TODO fix with proper modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import SEM, KerasLDS, LinearDynamicSystem, KerasMultiLayerPerceptron
from models import KerasSimpleRNN, KerasGRU
from opt.utils import evaluate, randstr
# end of imports -- this line is a total hack so we can get the imports for the jupyter notebooks. Pls don't remove

# lists of tests to run as (class name, pretraining params, params) pairs
# leave pretraining params empty for no pretraining
#
tests_to_run = [
#    ('Two2DGaussians', [2, 0.01], [2, 0.01]),
#    ('Two2DGaussians', [2, 0.01], [2, 0.01]),
#    ('Two2DGaussians', [2, 0.01], [2, 0.01]),
#    ('Two2DGaussians', [], [6, 0.01]),
#    ('TwoAlternating2DGaussians', [], [100, 0.01]),
#    ('TwoLinearDynamicalSystems', [], [100, 0.01]),
#    ('MotionCaptureData', [], [10]),
#    ('CoffeeShopWorldData', [], [20, 2, 400]),
    ('CoffeeShopWorldData', [2, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [2, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [2, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [2, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [2, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [2, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [4, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [4, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [4, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [4, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [4, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [4, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [6, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [6, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [6, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [6, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [6, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [6, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [8, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [8, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [8, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [8, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [8, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [8, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [10, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [10, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [10, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [10, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [10, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [10, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [15, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [15, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [15, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [15, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [15, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [15, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [20, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [20, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [20, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [20, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [20, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [20, 2, 400], [2, 2, 400]),

    ('CoffeeShopWorldData', [40, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [40, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [40, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [40, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [40, 2, 400], [2, 2, 400]),
    ('CoffeeShopWorldData', [40, 2, 400], [2, 2, 400]),
]

TOKEN = randstr(10) # random token for output filenames
notebook_filename = 'test_notebook_' + TOKEN + '.ipynb'

OUTPUT_DIR = 'output_' + TOKEN
os.makedirs(OUTPUT_DIR)

def get_results_filename(test_idx, suffix=''):
    if suffix:
        suffix = '_' + suffix
    filename = os.path.join(OUTPUT_DIR, 'test_results_' + TOKEN + '_' + str(test_idx) + suffix + '.pkl')
    return filename

# export dictionary to file
#
def save_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

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
    t_horizon = 5 # time horizon to consider
    
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
    beta = 0.15
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
def run_test(pretrain_data, test_data, test_idx):
    # set the parameters for the models
    # TODO because we use D to define the parameters, we cannot fully decouple the parameters from the data set. So we have a chicken or egg problem here. Ideally, the user should be able to pass Omega and test_data here and they should be independent. For now, they're not
    Omega = get_Omega(test_data.D)
  
    K = test_data.X.shape[0]
    sem = SEM(**Omega)

    # optionally pre-train the model in a supervised way using a different data set
    #
    if pretrain_data:
        #res = read_pickle('output/test_results_0YD0WL0DD2_0_pretrain.pkl')
        #sem.deserialize(res['sem'])
        #sem.k_prev = None
        #sem.x_prev = None

        sem.pretrain(pretrain_data.X, pretrain_data.y)

        sem.C[0] = 1000000000 # TODO FIXME NOTE rm -rf
        sem.C[1] = 1000000000
   
        # save results
        results_filename = get_results_filename(test_idx, 'pretrain')
        print 'Saving pretraining results to file ', results_filename
        res = {'pretrain_data': pretrain_data,
               'Omega': Omega,
               'sem': sem.serialize(OUTPUT_DIR)}
        save_pickle(res, results_filename)

    # run SEM on test data
    #
    post, pe, log_lik, log_prior = sem.run(test_data.X, K=K, return_pe=True, return_lik_prior=True)
    mi, r = test_data.performance(post)

    # save results
    results_filename = get_results_filename(test_idx)
    print 'Saving results to file ', results_filename
    res = {'test_data': test_data,
           'Omega': Omega,
           'sem': sem.serialize(OUTPUT_DIR),
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
    #
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

    # summary results
    #
    results_filename = get_results_filename(None, 'summary')
    code = ["# Show summary statistics\n",
            "\n",
            "res = read_pickle('" + results_filename + "')\n",
            "\n",
            "rs = res['rs']\n",
            "plt.plot(rs)\n",
            "plt.xlabel('test #')\n",
            "plt.ylabel('adjusted rand index')\n",
            "plt.show()\n"]
    nb_cells.append(new_code_cell(source=code))

    # tests
    #
    assert len(tests_to_run) == len(test_datas)
    for test_idx in range(len(tests_to_run)):
        test = tests_to_run[test_idx]
        pretrain_data = test_datas[test_idx][0]
        test_data = test_datas[test_idx][1]

        test_name = test[0]
        pretrain_params = test[1]
        pretrain_params_str = [str(x) for x in pretrain_params]
        test_params = test[2]
        test_params_str = [str(x) for x in test_params]

        # Load test data
        #
        title = '# Running test #' + str(test_idx) + ': ' + test_name + ' with params ' + ', '.join(test_params_str)
        if pretrain_params:
            title = title + ' with pretraining params ' + ', '.join(pretrain_params_str)
        title = title + '\n'
        nb_cells.append(new_markdown_cell(source=title))

        code = ['# Ensure reproducibility\n',
                '#\n',
                'np.random.seed(' + str(test_idx) + ')\n']
        nb_cells.append(new_code_cell(source=code))

        # it is important that these two are back to back -- otherwise, some np.random might sneak in between them
        # and screw up the test_data
        if pretrain_params:
            code = ['# Load pretrain data\n',
                    '# \n',
                    'pretrain_data = ' + test_name + '(' + ','.join(pretrain_params_str) + ')\n']
            nb_cells.append(new_code_cell(source=code))
        code = ['# Load test data\n',
                '# \n',
                'test_data = ' + test_name + '(' + ','.join(test_params_str) + ')\n']
        nb_cells.append(new_code_cell(source=code))

        if pretrain_params:
            code = ['# Plot pretrain data\n',
                    '# \n',
                    'pretrain_data.plot_scenes()\n']
            nb_cells.append(new_code_cell(source=code))

        code = ['# Plot test data\n',
                '# \n',
                'test_data.plot_scenes()\n']
        nb_cells.append(new_code_cell(source=code))

        code = get_code(get_Omega)
        code = ['# Set SEM parameters\n',
                '#\n',
                'D = test_data.D\n'] + code
        nb_cells.append(new_code_cell(source=code))

        # Init SEM 
        #
        code = ['# Initialize SEM\n',
                '# \n',
                'K = test_data.X.shape[0]\n',
                'sem = SEM(**Omega)\n']
        nb_cells.append(new_code_cell(source=code))

        if pretrain_params:
            # Optionally pretrain SEM
            #
            code = ['# Pretrain SEM\n',
                    '# \n',
                    '#sem.pretrain(pretrain_data.X, pretrain_data.y)\n']
            nb_cells.append(new_code_cell(source=code))

            # Alternative -- load pretrained sem from pickle file
            #
            results_filename = get_results_filename(test_idx, 'pretrain')
            code = ["# Alternatively, load pretrained SEM from past execution\n",
                    "#\n",
                    "res = read_pickle('" + results_filename + "')\n",
                    "sem.deserialize(res['sem'])\n"]
            nb_cells.append(new_code_cell(source=code))

        # Run SEM
        #
        code = ['# Run SEM\n',
                '# \n',
                '#post, pe, log_lik, log_prior = sem.run(test_data.X, K=K, return_pe=True, return_lik_prior=True)\n']
        nb_cells.append(new_code_cell(source=code))

        # Alternative -- load sem from pickle file
        #
        results_filename = get_results_filename(test_idx)
        code = ["# Alternatively, load results from past execution\n",
                "#\n",
                "res = read_pickle('" + results_filename + "')\n",
                "sem.deserialize(res['sem'])\n",
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

    # generate tests
    #
    for test_idx in range(len(tests_to_run)):
        test = tests_to_run[test_idx]
        test_name = test[0]
        pretrain_params = test[1]
        test_params = test[2]

        # for reproducibility -- note it's important NOT to call again before test_data (or it will generate identical pretrain and test data)
        # also note we should also call in notebook
        np.random.seed(test_idx)

        # generate test
        # it is important that these are back to back for reproducibility, lest some stray np.random
        # sneaks in between them and screws things up
        if pretrain_params:
            pretrain_data = getattr(sys.modules[__name__], test_name)(*pretrain_params)
        else:
            pretrain_data = None
        test_data = getattr(sys.modules[__name__], test_name)(*test_params)

        tests.append((pretrain_data, test_data)) # must do it before running them... otherwise it doesn't work

    # save jupyter notebook before running tests (as some of them may fail...)
    #
    write(tests_to_run, tests)

    # run tests
    #
    mis = []
    rs = []
    for test_idx in range(len(tests)):
        pretrain_data = tests[test_idx][0]
        test_data = tests[test_idx][1]
        print '\n\n      Running ', test_data, '\n\n'
        mi, r = run_test(pretrain_data, test_data, test_idx)
        mis.append(mi)
        rs.append(r)

    # save summary results
    #
    results_filename = get_results_filename(None, 'summary')
    print 'Saving summary results to file ', results_filename
    res = {'tests_to_run': tests_to_run,
           'tests': tests,
           'mis': mis,
           'rs': rs}
    save_pickle(res, results_filename)

