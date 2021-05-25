import pandas as pd

class Tokenizer():

    def __init__(self):
        pass

    def tokenize(self, s):
        tokens = s.split(' ')
        return tokens

def excel_iter(path):
    data = pd.read_excel('IR_Spring2021_ph12_7k.xlsx')
    for i, row in data.iterrows():
        idx = row['id'] 
        content = row['content']
        url = row['url']
        yield (idx, content, url)


class InvertedIndex():
    def __init__(self):
        self.index = {}
    
    def add_tokens(self, tokens, doc_id):
        for token in tokens:
            if token not in self.index:
                self.index[token] = [doc_id]
            else:
                if self.index[token][-1] != doc_id:
                    self.index[token].append(doc_id)
