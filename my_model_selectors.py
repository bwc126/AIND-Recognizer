import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Initialize some objects that will be used later.
        best_model = object
        # cur_model = object
        best_score = float('inf')
        word_sequences = self.sequences
        num_samples = len(word_sequences)
        # We need to restrict the number of Kfold splits to [2,3]
        # num_splits = max(min(len(word_sequences), 3), 2)
        # For each possible number of states in the model, we'll see how it performs
        for n_components in range(self.min_n_components, self.max_n_components+1):
            m = n_components
            f = self.X.shape[1]
            # split_method = KFold(num_splits)
            cur_scores = []
            # Use try/except to handle non-viability of some models
            try:
                # # We use the average score of several trials across multiple K-fold splits
                # for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                # Combine our training sequences
                # train_x, train_lengths = combine_sequences(cv_train_idx, word_sequences)
                # Combine our testing sequences
                # test_x, test_lengths = combine_sequences(cv_test_idx, word_sequences)
                # The current model we're testing
                cur_model = GaussianHMM(n_components).fit(self.X, self.lengths)
                # Calculate its score
                logL = cur_model.score(self.X, self.lengths)
                BIC = -2 * logL + np.log(num_samples) * (m**2 + 2*m*f - 1)
                # Track the scores for this model, for each fold
                cur_score = BIC
            except:
                break
            # If the current model sets a new record for performance, it's our new best model
            if cur_score < best_score:
                best_score = cur_score
                best_model = cur_model
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Initialize some objects that will be used later.
        best_model = self.base_model(self.min_n_components)
        best_score = float('inf')
        word_sequences = self.sequences
        num_samples = len(word_sequences)
        # cur_model = object
        # We need to restrict the number of Kfold splits to [2,3]
        # num_splits = max(min(len(word_sequences), 3), 2)
        # split_method = KFold(num_splits)
        M = len(self.hwords)
        # For each possible number of states in the model, we'll see how it performs
        for n_components in range(self.min_n_components, self.max_n_components+1):
            cur_scores = []
            # Use try/except to handle non-viability of some models
            try:
                # We use the average score of several trials across multiple K-fold splits
                # for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                # Combine our training sequences
                # train_x, train_lengths = combine_sequences(cv_train_idx, word_sequences)
                # Combine our testing sequences
                # test_x, test_lengths = combine_sequences(cv_test_idx, word_sequences)
                # The current model we're testing
                cur_model = GaussianHMM(n_components).fit(self.X, self.lengths)
                #logP(original word) - 1 / (num words -1) * sum (logP(all other words))
                logP = cur_model.score(self.X, self.lengths)
                # We need to sum the error for the words that are ostensibly not the ones we're training for, and compare how well the model discriminates between its target word and others.
                anti_likelihoods = 0.0
                words = [word for word in self.words if word != self.this_word]
                # Here's where we actually see how well it identifies the other words
                for word in words:
                    word_x, word_lengths = self.hwords[word]
                    anti_likelihoods += cur_model.score(word_x, word_lengths)
                                        # Calculate its score
                avg_anti_likelihood = anti_likelihoods / (M-1)
                # Calculate the difference between the model's ability to identify its target word compared to its ability to identify other words
                DIC = logP - avg_anti_likelihood
                # Track the scores for this model, for each fold
                cur_score = DIC
            except:
                break

            # If the current model sets a new record for performance, it's our new best model
            if cur_score < best_score:
                best_score = cur_score
                best_model = cur_model
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Initialize some objects that will be used later.
        best_model = object
        cur_model = object
        best_score = float('-inf')
        word_sequences = self.sequences
        # We need to restrict the number of Kfold splits to [2,3]
        num_splits = max(min(len(word_sequences), 3), 2)
        # For each possible number of states in the model, we'll see how it performs
        for n_components in range(self.min_n_components, self.max_n_components+1):
            split_method = KFold(num_splits)
            cur_scores = []
        # Use try/except to handle non-viability of some models
            try:
            # We use the average score of several trials across multiple K-fold splits
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    # Combine our training sequences
                    test_x, test_lengths = combine_sequences(cv_test_idx, word_sequences)
                    # Combine our testing sequences
                    train_x, train_lengths = combine_sequences(cv_train_idx, word_sequences)
                    # The current model we're testing
                    cur_model = GaussianHMM(n_components).fit(train_x, train_lengths)
                    # Track the scores for this model, for each fold
                    cur_scores.append(cur_model.score(test_x, test_lengths))
            except:
                break

            # If the current model sets a new record for performance, it's our new best model
            if np.mean(cur_scores) > best_score:
                best_score = np.mean(cur_scores)
                best_model = cur_model
        return best_model
