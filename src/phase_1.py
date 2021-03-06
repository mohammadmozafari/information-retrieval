import re
import itertools
import pandas as pd

def main():

    index = None
    stemmer = Stemmer()
    while True:
        print('------------------------------')
        print('1 - Create index')
        print('2 - Load index')
        print('3 - Search ...')
        print('4 - Exit')
        choice = int(input())
        print()
        if choice == 1:
            index = create_index('data.xlsx', 'index.txt', stemmer)
        elif choice == 2:
            index = load_index('index.txt')
        elif choice == 3:
            if index is None:
                print('You have to load the index first.')
                continue
            query = input('Enter search query: ')
            query = stemmer.stem(query)
            results = search(index, query)
            show_search_results(results, 'data.xlsx')
        elif choice == 4:
            break
        else:
            print('Invalid choice. Try again.')

def create_index(data_path, save_path, stemmer):
    inverted_index = InvertedIndex()
    tokenizer = Tokenizer()

    # Construct the inverted index
    for idx, content, _ in excel_iter(data_path):
        tokens = tokenizer.tokenize(content)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        inverted_index.add_tokens(stemmed_tokens, idx)
        if idx % 100 == 99:
            print('{} docs processed.'.format(idx + 1))

    # Eliminate the stop words
    inverted_index.remove_stop_words(freq_threshold=1000)
    inverted_index.save_in_file(save_path)
    
    return inverted_index


def load_index(index_path):
    inverted_index = InvertedIndex()
    inverted_index.load_from_file(index_path)
    return inverted_index


def search(index, query):
    words = query.split()
    if len(words) == 1:
        lst = index.get_postings_list(words[0])
        if lst is None:
            return None
        return [(x, None) for x in index.get_postings_list(words[0])]
    
    all_lists = []
    for word in words:
        pl = index.get_postings_list(word)
        if pl is not None:
            all_lists.append(pl)
    pointers = [0 for _ in range(len(all_lists))]
    results = []
    if len(pointers) == 0:
        return None
    while True:
        doc_ids = [all_lists[i][pointer] for (i, pointer) in enumerate(pointers)]
        doc_id, args = args_min(doc_ids)
        results.append((doc_id, len(args)))
        for arg in args:
            pointers[arg] += 1
        args = sorted(args, reverse=True)
        for arg in args:
            if pointers[arg] >= len(all_lists[arg]):
                pointers.pop(arg)
                all_lists.pop(arg)
        if len(pointers) == 0:
            break
    results = sorted(results, reverse=True, key=lambda x: x[1])
    return results

def args_min(lst):
    min_value = -1
    args = []
    for i, value in enumerate(lst):
        if (min_value == -1) or (value < min_value):
            min_value = value
            args = [i]
        elif value == min_value:
            args.append(i)
    return min_value, args

def show_search_results(doc_list, path):
    print()
    if doc_list is None:
        print('No results was found.')
        return
    data = pd.read_excel(path)
    for i, (doc_id, count) in enumerate(doc_list):
        print('{}.'.format(i + 1))
        print('\tDocument Number: {}'.format(doc_id))
        if count is not None:
            print('\tDocument contains {} words of query.'.format(count))
        print('\t{}'.format(data.loc[data['id'] == doc_id].iloc[0]['url']))

class Tokenizer():

    def __init__(self):
        pass

    def tokenize(self, s):
        normalized_s = self.normalize(s)
        tokens = normalized_s.split()
        return tokens

    def normalize(self, s):
        s = re.sub(r'[.????:??!????\-=%\[\]\(\)/]+\ *', ' ', s)
        s = re.sub(r'[.,:;?\(\)\[\]"\'!@#$%^&*\-/]', ' ', s)
        s = s.replace('\u200c', '')
        return s

def excel_iter(path):
    data = pd.read_excel(path)
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

    def save_in_file(self, path):
        with open(path, encoding='utf-8', mode='w') as f:
            for key, postings in self.index.items():
                f.write(key)
                for posting in postings:
                    f.write('\t{}'.format(posting))
                f.write('\n')

    def load_from_file(self, path):
        self.index = {}
        with open(path, encoding='utf-8') as f:
            while True:
                line = f.readline().strip()
                if line == '': break
                parts = line.split('\t')
                self.index[parts[0]] = [int(x) for x in parts[1:]]

    def get_postings_list(self, word):
        return self.index[word] if word in self.index else None

# Ideas: 
#   1 - When removing suffixes or prefixes check that the remainder has lenght more than 1
#   2 - When removing suffixes and prefixes from verbs check if the word is really a verb
class Stemmer():

    def __init__(self):
        self.suffixes = ['????' ,'????' ,'????' ,'????????']
        self.verb_prefixes = ['????', '??', '??????', '??']
        self.verb_suffixes = ['??', '??', '??', '????', '????', '????']
        self.erabs = ['??', '??', '??', '??', '??', '??', '??']
        self.bon_mazi = self.load_bon('./bon_mazi.fa')
        self.bon_mozare = self.load_bon('./bon_mozare.fa')
        self.mokassar_dic = self.load_mokassar('./Mokassar.fa')

    def load_bon(self, file):
        bons = set()
        with open(file, encoding='utf-8') as f:
            while True:
                line = f.readline().strip()
                if line == '':
                    break
                bons.add(line)
        return bons

    def load_mokassar(self, file):
        mokassar_dic = {}
        with open(file, encoding='utf-8') as f:
            while True:
                line = f.readline().strip()
                if line == '':
                    break
                mokassar, mofrad = line.split('\t')
                mokassar_dic[mokassar] = mofrad
        return mokassar_dic

    def remove_suffix(self, word):
        clean_word = word
        for suffix in self.suffixes:
            if clean_word.endswith(suffix):
                if len(clean_word[:-len(suffix)]) < 2:
                    continue
                clean_word = clean_word[:-len(suffix)]
                clean_word = self.remove_suffix(clean_word)
                break
        return clean_word

    def normalize_letters(self, word):
        clean_word = word
        x = r'??'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??|??|???|??'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|??'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|???|???'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|???|???|???|???'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|??|??|???|??|??|???|???|???|???'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|???'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|???'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r"???|??|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|??|??|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|??|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|??|??|??|??|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|??|???|???|??|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|??|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|??|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|???|???|???|???|??|???|???|???|??|??"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|??"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|???|???|???"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"??|???|??|??|???|??|??|??|??|??|???|???|??"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|??|???|???|???|??|??|??|??"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r"???|???|??|???|???|???|??|???|???|???|??|???|???|???|??|??|??|??"
        y = r"??"
        clean_word = re.sub(x, y, clean_word)
        x = r'??'
        y = r'???'
        clean_word = re.sub(x, y, clean_word)
        x = r'???|??|???|??|???|???|???|???'
        y = r'.'
        clean_word = re.sub(x, y, clean_word)
        x = r',|??|??|???|???'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'??'
        y = r'??'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'0'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'1'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'2'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'3'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'4'
        clean_word = re.sub(x, y, clean_word)
        x = r'??'
        y = r'5'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'6'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'7'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'8'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??'
        y = r'9'
        clean_word = re.sub(x, y, clean_word)
        x = r'??|??|??|??|??|??|??|'
        y = r''
        clean_word = re.sub(x, y, clean_word)
        x = r'( )+'
        y = r' '
        clean_word = re.sub(x, y, clean_word)
        x = r'(\n)+'
        y = r'\n'
        clean_word = re.sub(x, y, clean_word)
        return clean_word

    def normalize_if_verb(self, verb):
        if (verb in self.bon_mazi) or (verb in self.bon_mozare):
            return verb
        for pre, suf in itertools.product(self.verb_prefixes, self.verb_suffixes):
            if verb.startswith(pre):
                temp = verb[len(pre):]
                if (temp in self.bon_mazi) and (len(temp) > 1):
                    return temp
            if verb.endswith(suf):
                temp = verb[:-len(suf)]
                if ((temp in self.bon_mazi) or (temp in self.bon_mozare)) and (len(temp) > 1):
                    return temp
            if verb.startswith(pre) and verb.endswith(suf):
                temp = verb[len(pre):-len(suf)]
                if ((temp in self.bon_mazi) or (temp in self.bon_mozare)) and (len(temp) > 1):
                    return temp
        return verb

    def plural_to_single(self, word):
        if word in self.mokassar_dic:
            return self.mokassar_dic[word]
        return word

    def remove_erabs(self, word):
        clean_word = word
        for erab in self.erabs:
            clean_word = clean_word.replace(erab, '')
        return clean_word

    def stem(self, s):

        # Rule 1: Remove erabs
        s = self.remove_erabs(s)
        
        # Rule 2: Normalize letters in words
        s = self.normalize_letters(s)
        
        # Rule 3: Mokassar plurals
        s = self.plural_to_single(s)

        # Rule 4: Verbs are normalized
        s = self.normalize_if_verb(s)

        # Rule 5: Remove suffix from nouns
        s = self.remove_suffix(s) 

        return s

def test_tokenizer_and_stemmer():
    tk = Tokenizer()
    stemmer = Stemmer()
    data = pd.read_excel('data.xlsx')
    text = data.iloc[0]['content']
    tokens = tk.tokenize(text)
    for token in tokens:
        stemmed = stemmer.stem(token)
        if stemmed != token:
            print(token, stemmer.stem(token))


if __name__ == '__main__':
    
    # test_tokenizer_and_stemmer()
    main()
