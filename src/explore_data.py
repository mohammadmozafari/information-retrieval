import pandas as pd

def main():
    data = pd.read_excel('IR_Spring2021_ph12_7k.xlsx')
    char_set = set()
    for i, row in data.iterrows():
        content = row['content']
        char_set.update(set(content))
    print(char_set)

if __name__ == "__main__":
    main()