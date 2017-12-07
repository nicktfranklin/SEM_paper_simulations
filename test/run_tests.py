import numpy as np
import matplotlib.pyplot as plt

# test datasets
from toy_data import Two2DGaussians
from toy_data import TwoAlternating2DGaussians
from toy_data import TwoLinearDynamicalSystems
from motion_data import MotionCaptureData

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
    ('Two2DGaussians', [50, 0.01]),
    ('TwoAlternating2DGaussians', [100, 0.01]),
    ('TwoLinearDynamicalSystems', [100, 0.01]),
    ('MotionCaptureData', [10]),
]

def randstr(N=10):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

"""
Run a single test with given TestData object
"""
def run_test(test_data):
    # set the parameters for the models
    # TODO because we use D to define the parameters, we cannot fully decouple the parameters from the data set. So we have a chicken or egg problem here. Ideally, the user should be able to pass Omega and test_data here and they should be independent. For now, they're not
    D = test_data.D
    
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

    r, post, pe, log_like, log_prior = evaluate(test_data.X, test_data.y, Omega, save=False, return_pe=True, split_post=True)

    ami, rs = test_data.performance(post)
    return ami, rs


"""
Generate jupyter notebook for given tests
"""
def write(tests_to_run):
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    from nbformat import writes
    from inspect import getsourcelines

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
    for test in tests_to_run:
        test_name = test[0]
        test_params = test[1]
        test_params_str = [str(x) for x in test_params]

        # title
        title = '# Running test ' + test_name + ' with params ' + ', '.join(test_params_str) + '\n'
        nb_cells.append(new_markdown_cell(source=title))

        # test code
        code = getsourcelines(run_test)[0] # get run_test source
        code = code[1:] # strip f'n title
        code = [x[4:] for x in code] # remove indents 

        code = ['test_data = ' + test_name + '(' + ','.join(test_params_str) + ')\n',
                'test_data.plot_scenes()\n', '\n'] \
                + code

        # SEM initialization code
        init_code = [x for x in code if 'evaluate(' not in x and 'performance(' not in x and 'return' not in x]
        nb_cells.append(new_code_cell(source=init_code))

        # SEM run code
        eval_code = [x for x in code if 'evaluate(' in x]
        nb_cells.append(new_code_cell(source=eval_code))
    
        # performance code
        perf_code = [x for x in code if 'performance(' in x]
        nb_cells.append(new_code_cell(source=perf_code))

        # plotting code
        plot_code = ['test_data.plot_segmentation(post)\n']
        nb_cells.append(new_code_cell(source=plot_code))

    # create notebook
    nb = new_notebook(cells=nb_cells)
    filestr = writes(nb, version=4)

    # write notebook to file
    fname = 'test_notebook_' + randstr(10) + '.ipynb'
    print 'Saving as jupyter notebook to ', fname
    with open(fname, 'w') as f:
        f.write(filestr)


if __name__ == "__main__":
    tests = []

    # run tests
    for test in tests_to_run:
        test_name = test[0]
        test_params = test[1]
        test_data = getattr(sys.modules[__name__], test_name)(*test_params)
        tests.append(test_data) # must do it before running them... otherwise it doesn't work

    for test_data in tests:
        print '\n\n      Running ', test_data, '\n\n'
        ami, rs = run_test(test_data)

    # save jupyter notebook
    write(tests_to_run)

