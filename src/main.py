import pandas as pd
from utils import excel_iter, InvertedIndex, Tokenizer

def main():
    
    inverted_index = InvertedIndex()
    tokenizer = Tokenizer()

    # Construct the inverted index
    data_path = './IR_Spring2021_ph12_7k.xlsx'
    for idx, content, url in excel_iter(data_path):
        tokens = tokenizer.tokenize(content)
        inverted_index.add_tokens(tokens, idx)

    # Eliminate the stop words
    print(list(inverted_index.stop_words()))
    print(len(inverted_index.index))
    inverted_index.remove_stop_words(freq_threshold=1000)
    print(len(inverted_index.index))

if __name__ == "__main__":
    main()
