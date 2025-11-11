import re
import nltk

RE_SPACES = re.compile("\s+")
RE_HTTP = re.compile("http(s)?://[/\.a-z0-9]+")
RE_HASHTAG = re.compile("[@#][_a-z0-9]+")
RE_EMOTICONS = re.compile("(:-?\))|(:p)|(:d+)|(:-?\()|(:/)|(;-?\))|(<3)|(=\))|(\)-?:)|(:'\()|(8\))")


class Tokenizator():
    @staticmethod
    def tokenize(text):
        pass


class SimpleTokenizator(Tokenizator):
    @staticmethod
    def tokenize(text):
        return RE_SPACES.split(text.strip())

class BeforeTokenizationNormalizator():
    @staticmethod
    def normalize(text):
        text = text.strip().lower()

        text = text.replace('&nbsp;', u' ')
        text = text.replace('&lt;', u'<')
        text = text.replace('&gt;', u'>')
        text = text.replace('&amp;', u'&')
        text = text.replace('&pound;', u'£')
        text = text.replace('&euro;', u'€')
        text = text.replace('&copy;', u'©')
        text = text.replace('&reg;', u'®')
        return text

class AdvTokenizator(Tokenizator):
    @staticmethod
    def tokenize(text):
        tokens = SimpleTokenizator.tokenize(text)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            match = None
            for regexpr in [RE_HTTP, RE_HASHTAG, RE_EMOTICONS]:
                match = regexpr.search(token)
                if match is not None:
                    break
            if match is not None:
                idx_start, idx_end = match.start(), match.end()
                if idx_start != 0 or idx_end != len(token):
                    if idx_start != 0:
                        tokens[i] = token[:idx_start]
                        tokens.insert(i + 1, token[idx_start:])
                    else:
                        tokens[i] = token[:idx_end]
                        tokens.insert(i + 1, token[idx_end:])
                    i -= 1
            else:
                del tokens[i]
                tokens[i:i] = nltk.word_tokenize(token)
            i += 1
        return tokens



def tokenize_to_tweet_tokens(text):
    return AdvTokenizator.tokenize(BeforeTokenizationNormalizator.normalize(text))

    
class Tweet(object):
    def __init__(self, id, text, clazz):
        self.id = id
        self.text = text
        self.clazz = clazz

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

def tweets_in(path):
    for line in open(path, 'r'):
        line = line.strip().split('\t')
        assert len(line) == 3
        if line[-1] == 'Not Available':
            continue
        yield Tweet(int(line[0]), line[2], line[1])

class DataSet(object):
    def __init__(self, files):
        self._read_dataset(tweets_in, files)

    def _read_dataset(self, tweets_in, files):
        print ("Reading data set", files)
        self.tweets = []
        for file in files:
            for tweet in tweets_in(file):
                tweet.tokens = tokenize_to_tweet_tokens(tweet.text)
                self.tweets.append(tweet)
                
