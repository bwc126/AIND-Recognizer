import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_index, test_word in enumerate(test_set.wordlist):
        test_x, test_lengths = test_set.get_item_Xlengths(word_index)
        prob_dict = {}
        scores = []
        guess = None
        best_score = float('-inf')
        for word in models:
            model = models[word]
            try:
                score = model.score(test_x, test_lengths)
                scores.append(score)
            except:
                score = float('-inf')
                scores.append(score)
            prob_dict[word] = score
            if score > best_score:
                best_score = score
                guess = word

        probabilities.append(prob_dict)
        guesses.append(guess)


    return probabilities, guesses
