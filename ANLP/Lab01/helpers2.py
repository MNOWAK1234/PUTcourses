KIEDYŚ W NOTEBOOKU

from scipy.sparse import csr_matrix
from collections import defaultdict


class FeatureSet (object):
    def start_preprocess(self):
        pass

    def preprocess(self, tweet):
        pass

    def end_preprocess(self):
        pass

    def convert(self, tweet):
        pass

    def get_num_of_features(self):
        pass
    
class NGramFeautureSet(FeatureSet):
    def __init__(self, n, min_occurrence=5):
        self.ngrams = defaultdict(int)
        self.n = n
        self.min_occurecne = min_occurrence

    def preprocess(self, tweet):
        for ngram in self.find_ngrams(tweet.tokens_normalized, self.n):
            self.ngrams[ngram] += 1

    def end_preprocess(self):
        self.ngrams = set(k for k, v in self.ngrams.items() if v >= self.min_occurecne)

    def convert(self, tweet):
        features = {}
        for ngram in self.find_ngrams(tweet.tokens_normalized, self.n):
            if ngram in self.ngrams:
                features[ngram] = 1
        return features
    
    @staticmethod
    def find_ngrams( input_list, n):
        ngrams = zip(*[input_list[i:] for i in range(n)])
        return set(["_".join(i) for i in ngrams])
    
    def get_num_of_features(self):
        return len(self.ngrams)

class FeatureGenerator(object):
    def __init__(self):
        self.features_sets = {}

    def add_feature_set(self, name, feature_set):
        assert name not in self.features_sets
        self.features_sets[name] = feature_set

    def feed_features(self, tweets):
        for name in self.features_sets:
            self.features_sets[name].start_preprocess()

        for tweet in tweets:
            for name in self.features_sets:
                self.features_sets[name].preprocess(tweet)

        for name in self.features_sets:
            self.features_sets[name].end_preprocess()

    def convert_to_features(self, tweets):
        for tweet in tweets:
            features = {}
            for feature_name, feature_set in self.features_sets.items():
                features[feature_name] = feature_set.convert(tweet)
            yield features

    def get_num_of_features(self):
        num_of_features = {}
        for feature_name, feature_set in self.features_sets.items():
            num_of_features[feature_name] = feature_set.get_num_of_features()
        return num_of_features

    def get_feature_index(self, start_from = 0):
        index = {}
        idx = start_from
        for feature_name, feature_set in self.features_sets.items():
            index[feature_name] = {}
            for feature in feature_set.enumerate_fatures_names():
                index[feature_name][feature] = idx
                idx += 1
        return index, idx

    def create_matrix(self, tweets):
        row = []
        col = []
        data = []
        index, num_of_features = self.get_feature_index()

        labels = []
        for i, tweet_features in enumerate(self.convert_to_features(tweets)):
            labels.append(tweets[i].clazz)
            for feature_set_name, feature_set in tweet_features.items():
                for feature_name, feature_value in feature_set.items():
                    row.append(i)
                    col.append(index[feature_set_name][feature_name])
                    data.append(feature_value)
        return csr_matrix((data, (row, col)), shape=(len(tweets), num_of_features)), labels


feature_generator = FeatureGenerator()
feature_generator.feed_features(training_set.tweets)
feature_generator.add_feature_set('2-gram', NGramFeautureSet(n=2))
print ("Features {0}".format(feature_generator.get_num_of_features()))

#### start comment
print ("Convert dataset...")
train_sparse_repr, train_labels = feature_generator.create_matrix(training_set.tweets)
