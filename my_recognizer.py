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
    # Initialize our return objects
    probabilities = []
    guesses = []
    # We need to recognize each word in the test set
    for word_index, test_word in enumerate(test_set.wordlist):
        # We retrieve the test data for each word in the test set
        test_x, test_lengths = test_set.get_item_Xlengths(word_index)
        # We prepare to store the predicted probabilities for each word using the model trained for our target word, as well as the score and best guess.
        prob_dict = {}
        scores = []
        guess = None
        best_score = float('-inf')
        # We try each model to see what word it predicts, what its best guess is, as well as the probability it gives for each word
        for word in models:
            model = models[word]
            # Handle non-viable models with try/except
            try:
                # We score this model's ability to detect the target word
                score = model.score(test_x, test_lengths)
                scores.append(score)
            except:
                score = float('-inf')
                scores.append(score)
            # We store this model's probability for identifying this word
            prob_dict[word] = score
            # If we have a new best score, this model has more certainty than any previous model in identifying this word, and we have a new best guess for what the word is.
            if score > best_score:
                best_score = score
                guess = word
        # We store our predicted probabilities for each word occuring according to each model, and all of our best guesses. 
        probabilities.append(prob_dict)
        guesses.append(guess)


    return probabilities, guesses
