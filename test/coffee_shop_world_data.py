import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal as mvn
from numpy.fft import fft, ifft

from test_data import TestData


# hack to import narrative from parent and parent's parent directory
# TODO fix with proper modules
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from narrative.src.engine import main

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from opt import plate_formula

class CoffeeShopWorldData(TestData):
    """ 
    Coffee Shop World test case
    """
    def __init__(self, n_iterations=2, n_repeats=2, D=None):
        stories_kwargs = dict(
            mark_end_state=False,  # attach end_of_state, end_of_story marker
            attach_questions=False,  # attach question marker at the end of the state (e.g. Q_subject)
            gen_symbolic_states=False,  # GEN_SYMBOLIC_STATES = False
            attach_role_marker=False,  # ATTACH_ROLE_MARKER = False
            attach_role_maker_before=['Pronoun', 'Name', 'Pronoun_possessive', 'Pronoun_object'],
        )
        input_fnames = ['poetry', 'fight']
        n_input_files = len(input_fnames)
        names_concat = '_'.join(input_fnames)

        # generate some stories from Coffee Shop World
        # scenes is e.g. [ [(Ask, verb), (Tom, agent), (Charan, patient)],
        #        [(Answer, verb), (Charan, agent), (Tom, patient)],
        #        [(Punch, verb), (Tom, agent), (Charan, patient)] ]
        # events is e.g. [0 0 0 0 0 1 1 1 1 1 0 0 0 0]
        #
        rand_seed = np.random.randint(10000000)
        scenes, events = main(rand_seed, input_fnames, n_input_files, names_concat, n_iterations, n_repeats, write_to_files=False, stories_kwargs=stories_kwargs)

        print scenes[:20]
        print events[:20]

        # extract unique roles and fillers
        #
        role_library = set([b[1] for s in scenes for b in s])
        filler_library = set([b[0] for s in scenes for b in s])

        print 'Roles: ', role_library
        print 'Fillers: ', filler_library

        # figure out how many dimensions we need for HRR embeddings
        #
        n = len(role_library) + len(filler_library);     # vocabulary size
        k = 8;      # maximum number of terms to be combined
        err = 0.01; # error probability
        if D is None:
            # optionally pass D; it is useful e.g. for pretraining 
            D = plate_formula(n, k, err);

        # create a library of vectors
        #
        role_dict = {r: embed(1, D) for r in role_library}
        filler_dict = {f: embed(1, D) for f in filler_library}

        # embed the scenes
        #
        embedded_scenes = []
        for scene in scenes:
            bound = [encode(filler_dict[filler], role_dict[role]) for filler, role in scene]
            embedded_scene = add(bound)
            embedded_scenes.append(embedded_scene)

        embedded_scenes = np.vstack(embedded_scenes)
    
        # print some stats
        #
        r = role_dict[role_dict.keys()[0]]
        print "Role    mean = %.2f, std = %.2f" % (np.mean(r), np.std(r))
        f = filler_dict[filler_dict.keys()[0]]
        print "Filler  mean = %.2f, std = %.2f" % (np.mean(f), np.std(f))
        e = encode(f, r)
        print "Encoded mean = %.2f, std = %.2f" % (np.mean(e), np.std(e))

        print 'X dimensions: ', embedded_scenes.shape

        self.X = embedded_scenes
        self.y = np.array(events)
        self.D = self.X.shape[1]
        assert self.D == D

#
# some HRR stuff specific to this test
#

# generate new random vector(s) corresponding to symbol(s)
#
def embed(N, D):
    # N = # of vectors to generate
    # D = dimension of vector (= n in Plate's paper)
    #
    return mvn.rvs(mean = np.zeros(D), cov = np.eye(D) * 1/D, size=N)

# circular convolution c = a * b
#
def conv(a, b):
    return np.real(ifft(fft(a) * fft(b)))

# involution of a^* -- a_i = a_-i, modulo D (dimension of a)
#
def involution():
    return np.real(ifft(np.conj(fft(c))))

# circular correlation c = a # b = a^* * b
# approximately inverts circular convolution
# so that b ~= a # (a * b)
#
def corr(a, b):
    return np.real(ifft(np.conj(fft(a)) * fft(b)))

# bind filler a with role b
#
def encode(a, b):
    return conv(b, a) # swap them to confuse everybody

# add a list of HRRs. Makes sure to divide by sqrt(size of list)
#
def add(hrrs):
    return np.sum(hrrs, axis=0) / np.sqrt(len(hrrs))

# recover filler with role b from vector a
#
# Notice that I'm not dividing by length(a)!
# Sam does that because of his "spike" in the embeddigns which increases the variance
#
def decode(a, b):
    return corr(b, a) # swapped again

# HRR example from Sam's code
#
def hrr_example():
    D = 200

    dog = embed(1, D)
    agent = embed(1, D)
    cat = embed(1, D)
    patient = embed(1, D)
    chase = embed(1, D)
    verb = embed(1, D)

    #sentence = (encode(dog,agent) + encode(chase,verb) + encode(cat,patient)) / np.sqrt(3)
    sentence = add([encode(dog,agent), encode(chase,verb), encode(cat,patient)])
    print sentence
    dog_decoded = decode(sentence,agent)

    plt.plot(dog)
    plt.plot(sentence)
    plt.plot(dog_decoded)
    plt.show()

    print "dog    mean = %.2f, std = %.2f" % (np.mean(dog), np.std(dog))
    print "sentence    mean = %.2f, std = %.2f" % (np.mean(sentence), np.std(sentence))
    print "dog_decoded_decoded    mean = %.2f, std = %.2f" % (np.mean(dog_decoded), np.std(dog_decoded))

    print 'correlation w/ dog ', np.corrcoef(dog, dog_decoded)[0][1]
    print 'correlation w/ agent ', np.corrcoef(agent, dog_decoded)[0][1]
    print 'correlation w/ cat ', np.corrcoef(cat, dog_decoded)[0][1]
    print 'correlation w/ patient ', np.corrcoef(patient, dog_decoded)[0][1]
    print 'correlation w/ chase ', np.corrcoef(chase, dog_decoded)[0][1]
    print 'correlation w/ verb ', np.corrcoef(verb, dog_decoded)[0][1]
    print 'correlation w/ sentence ', np.corrcoef(sentence, dog_decoded)[0][1]
