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


