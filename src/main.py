import pandas as pd
from utils import excel_iter, Tokenizer

def main():

    tokenizer = Tokenizer()
    data_path = './IR_Spring2021_ph12_7k.xlsx'
    for idx, content, url in excel_iter(data_path):
        print(tokenizer.tokenize(content))
        break

if __name__ == "__main__":
    main()
