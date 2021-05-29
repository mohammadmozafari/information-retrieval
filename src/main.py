import pandas as pd
from utils import excel_iter, InvertedIndex, Tokenizer

def main():
    
    inverted_index = InvertedIndex()
    tokenizer = Tokenizer()

    data_path = './IR_Spring2021_ph12_7k.xlsx'
    for idx, content, url in excel_iter(data_path):
        tokens = tokenizer.tokenize(content)
        print(tokens)
        break
        inverted_index.add_tokens(tokens, idx)

if __name__ == "__main__":
    main()
