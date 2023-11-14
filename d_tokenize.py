from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

class Tokenizer():
    def __init__(self) -> None:
        nltk.download("punkt")
        nltk.download('stopwords')
        with open("stopwords.txt") as f:
            custom_stopwords = [line.strip() for line in f]
        self.all_stopwords = set(stopwords.words("english")).union(set(custom_stopwords))

    def tokenize_text(self, text):
        '''
        Tokenize title or description
        '''
        tokens = word_tokenize(text.lower())
        return [x for x in tokens if x not in self.all_stopwords and len(x) > 1]
    
    def tokenize_tag(self, tag_str):
        '''
        Tokenize tags string e.g. "cgpgrey|education|hello internet"
        '''
        if len(tag_str) > 0 and tag_str != "[None]":
            return [x.strip() for x in tag_str.split("|")]
        else:
            return []

