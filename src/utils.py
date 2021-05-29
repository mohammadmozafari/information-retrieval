import re
import itertools
import pandas as pd

class Tokenizer():

    def __init__(self):
        pass

    def tokenize(self, s):
        normalized_s = self.normalize(s)
        tokens = normalized_s.split()
        return tokens

    def normalize(self, s):
        s = re.sub(r'[.،؛:؟!«»\-=%\[\]\(\)/]+\ *', ' ', s)
        s = re.sub(r'[.,:;?\(\)\[\]"\'!@#$%^&*\-/]', ' ', s)
        s = s.replace('\u200c', '')
        return s

def excel_iter(path):
    data = pd.read_excel('IR_Spring2021_ph12_7k.xlsx')
    for _, row in data.iterrows():
        idx = row['id'] 
        content = row['content']
        url = row['url']
        yield (idx, content, url)


class InvertedIndex():
    """
    Represents a simple inverted index 
    that is stored in a python dictionary.
    """
    
    def __init__(self):
        self.index = {}

    def __str__(self) -> str:
        result = ''
        for i, (key, val) in enumerate(self.index.items()):
            result += '{} -> {} -> {} docs\n'.format(i + 1, key, len(val))
        return result
    
    def add_tokens(self, tokens, doc_id):
        for token in tokens:
            if token not in self.index:
                self.index[token] = [doc_id]
            else:
                if self.index[token][-1] != doc_id:
                    self.index[token].append(doc_id)

    def remove_stop_words(self, freq_threshold=1000):
        stop_words = list(self.stop_words(freq_threshold))
        print('number of stop words ', len(stop_words))
        for word in stop_words:
            del self.index[word]

    def stop_words(self, freq_threshold=1000):
        for key, val in self.index.items():
            if len(val) >= freq_threshold:
                yield key

class Stemmer():

    def __init__(self):
        self.suffixes = ['ها', 'ات', 'تر', 'ترین', 'آسا' ,'اسا' , 'سان']

    def removeSuffix(self, word):
        clean_word = word
        for suffix in self.suffixes:
            if clean_word.endswith(suffix):
                clean_word = clean_word[:-len(suffix)]
                clean_word = self.removeSuffix(clean_word)
                break
        return clean_word

    def stem(self, s):

        # first rule: remove suffix from nouns
        s = self.removeSuffix(s) 
        return s


if __name__ == '__main__':
    
    test_stem = Stemmer()
    new = test_stem.stem('برقآساترین')
    print(new)
