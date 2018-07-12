import numpy as np
import tensorflow as tf
import pandas as pd

from models import KerasGRU, KerasSimpleRNN, KerasLSTM, KerasGRU_stacked
from opt.csw_utils import parse_story, parser_fight, parser_poetry
from opt import embed, encode
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout




# enumerate the words and properties used in the stories. Note: the parser
# take care of the verbs (which correspond 1-to-1 with nodes)
agents = {
    'Olivia': ('sad', 'hungry', 'Female'),
    'Will': ('happy', 'hungry', 'Male'),
    'Mariko': ('nervous', 'not-hungry', 'Female'),
    'Pradeep': ('happy', 'not-hungry', 'Male'),
    'Sarah': ('nervous', 'hungry', 'Female'),
    'Julian': ('sad', 'not-hungry', 'Male'),
    'Oona': ('violent', 'not-thirsty', 'Female'),
    'Jesse': ('violent', 'thirsty', 'Female'),
    'Nick': ('not-violent', 'not-thirsty', 'Male'),
    'Vera': ('not-violent', 'thirsty', 'Female'),
    'Silvia': ('not-violent', 'not-thirsty', 'Female'),
    'Ben': ('violent', 'not-thirsty', 'Male'),
}

drinks = {
    'coffee': "$3",
    'tea': "$2",
    'latte': "$5",
    'mocha': None
}

dessert = {
    'cake': 'display_case',
    'sorbet': 'freezer',
    'mousse': 'refrigerator',
    'cookie': None,
    'lemonsquare': None,
    'muffin': None
}

# pull the raw text from the list of stories
stories_poetry = []
with open('poetry_1000_1.txt', 'r') as f:
    for line in f:

        # only parse a story without that isn't a blank line
        if line != '\n':
            stories_poetry.append(parse_story(line, parser_poetry))

stories_fight = []
with open('fight_mod_1000_1.txt', 'r') as f:
    for line in f:
        # only parse a story without that isn't a blank line
        if line != '\n':
            stories_fight.append(parse_story(line, parser_fight))

# get a list of all of the words for vector embeddings
words = set([w for s in stories_poetry for n in s for w in n])
for w in set([w for s in stories_fight for n in s for w in n]):
    words.add(w)
words.remove(None)
words.add('None')

# get a list of all of the properties
properties = set([v[0] for v in agents.values()] + [v[1] for v in agents.values()] + [v[2] for v in agents.values()] + \
                 [v for v in drinks.values()] + [v for v in dessert.values()])
# add special properties
properties.add('isPerson')
properties.add('isDrink')
properties.add('isDessert')
properties.add('isVerb')

# use plate's formula to determine the necessary dimensionality
n = len(words) + 3
k = 6;  # maximum number of terms to be combined
err = 0.05;  # error probability

# d = plate_formula(n, k, err);
d = 100
print "Embedding in %d Dimensions" % d

# embed the properies
properties_embedding = {
    w: embed(1, d) for w in properties
}

# embed the words
word_embeddings = {}
from sklearn.preprocessing import normalize

for w in words:
    embedding = embed(1, d)  # have a unique random vector for each word

    # build in the similarity structure
    if w in agents.keys():
        embedding += properties_embedding[agents[w][0]] + properties_embedding[agents[w][1]] + \
                     properties_embedding[agents[w][2]]
        embedding += properties_embedding['isPerson']

    elif w in drinks.keys():
        embedding += properties_embedding[drinks[w]] + properties_embedding['isDrink']

    elif w in dessert.keys():
        embedding += properties_embedding[dessert[w]] + properties_embedding['isDessert']

    else:
        embedding += properties_embedding['isVerb']

    # normalize the embedding
    word_embeddings[w] = normalize(embedding)

# fillers -- we don't have words for this put it is done by position in the node
# filler_embeddings = [embed_onehot(1, d) for _ in range(3)]
filler_embeddings = [embed(1, d) for _ in range(3)]


# Were going to present SEM with the first N stories and ask it to remember what people ordered
def embed_story_hrr(story):
    # encode each node in a story as a vector
    X = np.zeros((0, d))  # append t zero vectors at the beginning of each story
    for node in story:
        X_t = np.zeros((1, d))
        for p, w in enumerate(node):
            if w is not None:
                X_t += encode(word_embeddings[w], filler_embeddings[p])
            else:
                X_t += encode(word_embeddings['None'], filler_embeddings[p])
        X = np.concatenate([X, normalize(X_t)], axis=0)
    return X


def embed_story_bow(story):
    # encode each node in a story as a vector
    X = np.zeros((0, d))  # append t zero vectors at the beginning of each story
    y = []
    for node in story:
        X_t = np.zeros((1, d))
        for p, w in enumerate(node):
            if w is not None:
                X_t += word_embeddings[w]
            else:
                X_t += word_embeddings['None']
        X = np.concatenate([X, normalize(X_t)], axis=0)
    return X


def make_embedded_fight(n_stories, start=0):
    list_stories = []
    e_token, t = [], 0
    trials = np.zeros(0, dtype=int)
    for ii in range(start, n_stories + start):
        list_stories += stories_fight[ii]
        e_token += [t] * len(stories_fight[ii])
        trials = np.concatenate([trials, np.arange(1, len(stories_fight[ii]) + 1, dtype=int)])
        t += 1

    X = embed_story_hrr(list_stories)
    return X, e_token, list_stories, trials


def make_embedded_poetry(n_stories, start=0):
    list_stories = []
    e_token, t = [], 0
    trials = np.zeros(0, dtype=int)
    for ii in range(start, n_stories + start):
        list_stories += stories_poetry[ii]
        e_token += [t] * len(stories_poetry[ii])
        trials = np.concatenate([trials, np.arange(1, len(stories_poetry[ii]) + 1, dtype=int)])
        t += 1

    X = embed_story_hrr(list_stories)
    return X, e_token, list_stories, trials


def make_ap_flip_fight(n_stories, start=0):
    list_stories = []
    e_token, t = [], 0
    trials = np.zeros(0, dtype=int)
    for ii in range(start, n_stories + start):
        list_stories += stories_fight[ii]
        e_token += [t] * len(stories_fight[ii])
        trials = np.concatenate([trials, np.arange(1, len(stories_fight[ii]) + 1, dtype=int)])
        t += 1

    # flip the AP roles
    list_stories = [(v, p, a) for (v, a, p) in list_stories]

    X = embed_story_hrr(list_stories)
    gramatical = np.array([p in agents.keys() for (_, p, _) in list_stories])
    return X, gramatical


def make_ap_flip_poetry(n_stories, start=0):
    list_stories = []
    e_token, t = [], 0
    trials = np.zeros(0, dtype=int)
    for ii in range(start, n_stories + start):
        list_stories += stories_poetry[ii]
        e_token += [t] * len(stories_poetry[ii])
        trials = np.concatenate([trials, np.arange(1, len(stories_poetry[ii]) + 1, dtype=int)])
        t += 1

    # flip the AP roles
    list_stories = [(v, p, a) for (v, a, p) in list_stories]

    X = embed_story_hrr(list_stories)
    gramatical = np.array([p in agents.keys() for (_, p, _) in list_stories])
    return X, gramatical

def train_evaluate_event_model(X, event_tokens, f_class, f_opts):
    event_model = f_class(d, **f_opts)
    event_model.update_f0(X[0, :])
    y_pred_one_step = []
    # event_boundary = False
    y_pred_one_step.append(event_model._predict_f0())
    for ii in range(1, np.shape(X)[0]):
        if event_tokens[ii] != event_tokens[ii - 1]:
            y_pred_one_step.append(event_model._predict_f0())
            event_model.new_token()
            event_model.update_f0(X[ii, :])
        else:
            _X = X[(np.array(event_tokens) == event_tokens[ii]) & (np.arange(X.shape[0]) < ii), :]
            y_pred_one_step.append(event_model.predict_next_generative(_X))
            event_model.update(X[ii - 1, :], X[ii, :])

    events_dict = {e0: event_model for e0 in event_tokens}
    return events_dict, np.reshape(y_pred_one_step, X.shape)


class KerasGRU_s(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()

        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model = Sequential()
        self.model.add(GRU(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)


def main():
    # randomly draw starting points!
    n_batch = 20

    np.random.seed(1234)

    starts = np.random.randint(0, 900, n_batch)
    n_stories = 10

    def batch_model(f_class, f_opts):
        df = []
        for s in tqdm(starts, desc='Batches'):
            X_fight, e_fight, list_fight, trial_fight = make_embedded_fight(n_stories, start=s)
            X_poetr, e_poetr, list_poetr, trial_poetr = make_embedded_poetry(n_stories, start=s)

            _, y_pred_fight = train_evaluate_event_model(X_fight, e_fight, f_class, f_opts=f_opts)
            _, y_pred_poetry = train_evaluate_event_model(X_poetr, e_poetr, f_class, f_opts=f_opts)

            X_fight_apflip, gramatical_fight = make_ap_flip_fight(n_stories, start=s)
            X_poetr_apflip, gramatical_poetr = make_ap_flip_poetry(n_stories, start=s)

            df.append(pd.DataFrame({
                'Error': np.linalg.norm(y_pred_fight - X_fight, axis=1),
                'Error flip': np.linalg.norm(y_pred_fight - X_fight_apflip, axis=1),
                'Event': e_fight,
                'Trial': trial_fight,
                'Type': ['Fight'] * len(e_fight),
                'Batch': [s] * len(e_fight),
                'Gramatical Flip': gramatical_fight
            }))
            df.append(pd.DataFrame({
                'Error': np.linalg.norm(y_pred_poetry - X_poetr, axis=1),
                'Error flip': np.linalg.norm(y_pred_poetry - X_poetr_apflip, axis=1),
                'Event': e_poetr,
                'Trial': trial_poetr,
                'Type': ['Poetry'] * len(e_poetr),
                'Batch': [s] * len(e_poetr),
                'Gramatical Flip': gramatical_poetr
            }))

        return pd.concat(df)

    print "Running SRN"
    f_class = KerasSimpleRNN
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.00)
    df_SRN = batch_model(f_class, f_opts)

    print "Running GRU_s"
    f_class = KerasGRU_s
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, l2_regularization=0.0, dropout=0.10)
    df_GRU_s = batch_model(f_class, f_opts)

    print "Running GRU wDO"
    f_class = KerasGRU
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.10)
    df_GRU_do = batch_model(f_class, f_opts)

    print "Running GRU wDO2"
    f_class = KerasGRU
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.20)
    df_GRU_do2 = batch_model(f_class, f_opts)

    print "Running GRU small hidden"
    f_class = KerasGRU
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d / 2, n_hidden2=d / 2, l2_regularization=0.0, dropout=0.10)
    df_GRU_do3 = batch_model(f_class, f_opts)

    print "Running GRU small hidden"
    f_class = KerasGRU
    f_opts = dict(n_epochs=250, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.10)
    df_GRU_do4 = batch_model(f_class, f_opts)

    print "Running GRU stacked"
    f_class = KerasGRU_stacked
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.00)
    df_GRU_stacked = batch_model(f_class, f_opts)

    print "Running GRU"
    f_class = KerasGRU
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.00)
    df_GRU = batch_model(f_class, f_opts)

    print "Running GRU stacked wDO"
    f_class = KerasGRU_stacked
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.10)
    df_GRU_stacked_do = batch_model(f_class, f_opts)

    print "Running LSTM"
    f_class = KerasLSTM
    f_opts = dict(n_epochs=25, optimizer='adam', n_hidden1=d, n_hidden2=d, l2_regularization=0.0, dropout=0.00)
    df_LSTM = batch_model(f_class, f_opts)

    df_SRN['EventModel'] = 'SRN'
    df_GRU_s['EventModel'] = 'GRU-simple'
    df_GRU['EventModel'] = 'GRU'
    df_GRU_do['EventModel'] = 'GRU wDO=0.1'
    df_GRU_do2['EventModel'] = 'GRU wDO=0.2'
    df_GRU_do3['EventModel'] = 'GRU wDO=0.1, small layers'
    df_GRU_do4['EventModel'] = 'GRU wDO=0.1, batch=250'
    df_GRU_stacked['EventModel'] = 'GRU-Stacked'
    df_GRU_stacked_do['EventModel'] = 'GRU-Stacked wDO'
    df_LSTM['EventModel'] = 'LSTM'

    df = pd.concat([
        df_SRN,
        df_LSTM,
        df_GRU,
        df_GRU_do,
        df_GRU_do2,
        df_GRU_do3,
        df_GRU_do4,
        df_GRU_stacked,
        df_GRU_stacked_do,
        df_GRU_s, ])
    df['Un-Normed log Prob'] = -df.Error
    df.to_pickle('event_model_eval_res.pkl')