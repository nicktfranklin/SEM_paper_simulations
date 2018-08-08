import numpy as np
from opt.hrr import conv_circ


def sample_pmf(pmf):
    return np.sum(np.cumsum(pmf) < np.random.uniform(0, 1))


class Story(object):
    def __init__(self):
        self.states = dict()
        self.transition_function = np.zeros((len(self.states), len(self.states)))
        self.filler_embeddings = dict()

    def sample_trajectory(self):
        s = [0]
        while s[-1] < 7:
            s.append(sample_pmf(self.transition_function[s[-1], :]))
        return s

    def get_valid_sucessors(self, s):
        return np.arange(len(self.states))[self.transition_function[0, :] > 0]

    def evaluate_goodness(self, s):
        lg = 0.0

        reverse_key = {v: k for k, v in self.states.iteritems()}

        for ii in range(len(s)-1):
            s_0 = reverse_key[s[ii][:s[ii].find('(')]]
            s_1 = reverse_key[s[ii+1][:s[ii+1].find('(')]]
            lg += np.log(self.transition_function[s_0, s_1])
        return np.exp(lg)

    def embed_story(self, story_list):
        pass


class Wedding(Story):

    def __init__(self, d=25, embedding_seed=0):
        Story.__init__(self)

        self.d = d

        # list of states
        self.states = {
            0: 'Enter',
            1: 'MakeCampfire',
            2: 'PlantFlower',
            3: 'DropInBowl',
            4: 'HoldTorch',
            5: 'BreakAnEgg',
            6: 'MakeAPainting',
            7: 'LookAtGifts',
        }

        self.transition_function = np.zeros((len(self.states), len(self.states)))

        # state similarities -- i.e. encoding time as a feature of the states
        state_sims = {
            0: np.random.randn(1, d),
            1: np.random.randn(1, d),
            3: np.random.randn(1, d),
            5: np.random.randn(1, d),
            7: np.random.randn(1, d),
        }
        state_sims[2] = state_sims[1]
        state_sims[4] = state_sims[3]
        state_sims[6] = state_sims[5]

        # list the potential fillers
        self.female_names = ['Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Amelia', 'Evelyn',
                             'Abigail']
        self.male_names = ['Liam', 'Noah', 'William', 'James', 'Logan', 'Benjamin', 'Mason', 'Elijah', 'Oliver', 'Jacob']

        self.state_names = [v for v in self.states.itervalues()]

        # role tags -- includes the properties male and female
        self.word_tags = ['female', 'male', 'character', 'verb', 'egg_design', 'painting_design', 'gift']

        # these are just added when needed
        self.egg_designs = ['egg_' + n for n in 'one two three four five'.split()]
        self.painting_designs = ['painting_' + n for n in 'one two three four five'.split()]
        self.gifts = ['gift_' + n for n in 'one two three four five six seven eight nine ten'.split()]

        # define an embedding dictionary
        np.random.seed(embedding_seed)  # fix the seed for repeatability

        self.role_embeddings = {w: np.random.randn(1, d) for w in self.word_tags}

        self.filler_embeddings = {
            w:
                conv_circ(np.random.randn(1, d) + self.role_embeddings['female'], self.role_embeddings['character'], n=d)
            for w in self.female_names
        }

        self.filler_embeddings.update({
            w:
                conv_circ(np.random.randn(1, d) + self.role_embeddings['male'], self.role_embeddings['character'], n=d)
            for w in self.male_names
        })

        # go ahead and bind the verb?
        self.filler_embeddings.update({
            w:
                conv_circ((np.random.randn(1, d) + state_sims[s]), self.role_embeddings['verb'], n=d)
            for s, w in self.states.iteritems()})

        self.filler_embeddings.update({
            w:
                conv_circ(np.random.randn(1, d), self.role_embeddings['egg_design'], n=d)
            for w in self.egg_designs
        })
        self.filler_embeddings.update({
            w:
                conv_circ(np.random.randn(1, d), self.role_embeddings['painting_design'], n=d)
            for w in self.painting_designs
        })

        # these can't be pre-embedded --> update: maybe they can be?
        self.filler_embeddings.update({
            w:
                conv_circ(np.random.randn(1, d), self.role_embeddings['gift'], n=d)
            for w in self.gifts
        })

        # scale the variance of the features of the embedding space
        self.filler_embeddings = {k: v / np.sqrt(2. * d) for k, v in self.filler_embeddings.iteritems()}

        np.random.seed()  # reset the random seed now that the embeddings are drawn

        # make a word list for all of the embeddings
        self.words = self.filler_embeddings.keys()

        # define the same sex wedding probability
        self.same_sex_wedding_prob = 0.1

    def sample_story(self):
        s = self.sample_trajectory()

        # sample the fillers
        egg_design = self.egg_designs[np.random.randint(len(self.egg_designs))]
        painting_design = self.painting_designs[np.random.randint(len(self.painting_designs))]
        idx = range(len(self.gifts))
        np.random.shuffle(idx)
        gift_a = self.gifts[idx.pop()]
        gift_b = self.gifts[idx.pop()]

        char_sets = [[w for w in self.female_names], [w for w in self.male_names]]
        np.random.shuffle(char_sets)
        for cs in char_sets:
            np.random.shuffle(cs)
        charetor_a = char_sets[0].pop()
        if np.random.uniform(0, 1) < self.same_sex_wedding_prob:
            charetor_b = char_sets[0].pop()
        else:
            charetor_b = char_sets[1].pop()

        # generate the sequence of embedded vectors
        X = []
        # characters = conv_circ(self.filler_embeddings[charetor_a], self.filler_embeddings[charetor_b])
        characters = self.filler_embeddings[charetor_a] + self.filler_embeddings[charetor_b]
        for s0 in s:
            X.append(self.filler_embeddings[self.states[s0]] + characters)
            if s0 == 5:
                X[-1] += self.filler_embeddings[egg_design]
            if s0 == 6:
                X[-1] += self.filler_embeddings[painting_design]
            if s0 == 7:
                # X[-1] += conv_circ(self.filler_embeddings[gift_a], self.filler_embeddings[gift_b])
                X[-1] += self.filler_embeddings[gift_a] + self.filler_embeddings[gift_b]

        # generate the human-readable symbolic sentences
        sentences =[]
        for s0 in s:
            sentences.append( self.state_names[s0] + '(' + charetor_a + ', ' + charetor_b + ')')
            if s0 == 5:
                sentences[-1] = sentences[-1][:-1] + ', ' + egg_design + ')'
            if s0 == 6:
                sentences[-1] = sentences[-1][:-1] + ', ' + painting_design + ')'
            if s0 == 7:
                sentences[-1] = sentences[-1][:-1] + ', ' + gift_a + ', ' + gift_b + ')'
        return np.reshape(X, newshape=(-1, self.d,)), sentences

    def generate_foil_events(self, sentences):
        """

        :param sentences: a list of structured events (e.g. "Verb(Char1, Char2)", etc.)
        :return: a list of all possible next successor states, in embedded vector space
        """

        reverse_state_key = {w: k for k, w in self.state_names.iteritems()}

        for state in sentences:
            verb = state[:state.find('(')]
            idx = reverse_state_key[verb]

            # pull the successor states
            self.get_valid_sucessors(idx)

            #


class GreenWedding(Wedding):
    def __init__(self):
        Wedding.__init__(self)

        # hand craft the transition function
        self.transition_function[0, 1] = 0.8  # make campfire
        self.transition_function[0, 2] = 0.2  # plant flower

        self.transition_function[1, 3] = 0.8  # campfire -> drop in bowl
        self.transition_function[1, 4] = 0.2  # campfire -> hold torch

        self.transition_function[2, 3] = 0.8  # plant flower -> drop in bowl
        self.transition_function[2, 4] = 0.2  # plant flower -> hold torch

        self.transition_function[3, 5] = 0.8  # drop in bowl -> break egg
        self.transition_function[3, 6] = 0.2  # drop in bowl -> make a painting

        self.transition_function[4, 5] = 0.8  # hold torch -> break an egg
        self.transition_function[4, 6] = 0.2  # hold torch -> make a painting

        self.transition_function[5, 7] = 1.0  # break an egg -> Look at gifts
        self.transition_function[6, 7] = 1.0  # make a paiting -> look at gifts


class YellowWedding(Wedding):
    def __init__(self):
        Wedding.__init__(self)

        # hand craft the transition function
        self.transition_function[0, 1] = 0.2  # make campfire
        self.transition_function[0, 2] = 0.8  # plant flower

        self.transition_function[1, 3] = 0.2  # campfire -> drop in bowl
        self.transition_function[1, 4] = 0.8  # campfire -> hold torch

        self.transition_function[2, 3] = 0.2  # plant flower -> drop in bowl
        self.transition_function[2, 4] = 0.8  # plant flower -> hold torch

        self.transition_function[3, 5] = 0.2  # drop in bowl -> break egg
        self.transition_function[3, 6] = 0.8  # drop in bowl -> make a painting

        self.transition_function[4, 5] = 0.2  # hold torch -> break an egg
        self.transition_function[4, 6] = 0.8  # hold torch -> make a painting

        self.transition_function[5, 7] = 1.0  # break an egg -> Look at gifts
        self.transition_function[6, 7] = 1.0  # make a paiting -> look at gifts
