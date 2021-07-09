import re
import math
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
            show_top_k_results(results, 'data.xlsx')
        elif choice == 4:
            break
        else:
            print('Invalid choice. Try again.')


def create_index(data_path, save_path, stemmer):
    inverted_index = InvertedIndex()
    tokenizer = Tokenizer()

    # Construct the inverted index
    for idx, content, _ in excel_iter(data_path):
        inverted_index.num_docs += 1
        inverted_index.doc_lengths[idx] = len(content)
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
    """
    This function takes a query, finds its vector representation and
    calculates its cosine similarity with documents and returns document scores.
    """
    results = {}
    query_vector = compute_query_vector(index, query)
    for word, weight in query_vector.items():
        p = index.get_postings_list(word)
        if p is None:
            continue
        idf = math.log10(index.num_docs / len(index.index[word]))
        for doc, f in p:
            tf = 1 + math.log10(f)
            score = tf * idf * weight
            if doc in results:
                results[doc] += score
            else:
                results[doc] = score
    for result in results.keys():
        results[result] /= index.doc_lengths[result]
    return results

def compute_query_vector(index, query):
    """
    Computes tf.idf scores of each word in the query.
    """
    words = query.split()
    vector = {}
    for word in words:
        if word in vector:
            continue
        print(index.num_docs)
        print(len(index.index[word]))
        idf = math.log10(index.num_docs / len(index.index[word]))
        tf = 1 + math.log10(words.count(word))
        vector[word] = tf * idf
    return vector

def show_top_k_results(doc_list, path):
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


def excel_iter(path):
    data = pd.read_excel(path)
    for _, row in data.iterrows():
        idx = row['id']
        content = row['content']
        url = row['url']
        yield (idx, content, url)

# --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Inverted Index -----------------------------------------------------------

class InvertedIndex():
    """
    Represents a simple inverted index 
    that is stored in a python dictionary.
    """

    def __init__(self):
        self.index = {}
        self.doc_lengths = {}
        self.num_docs = 0

    def add_tokens(self, tokens, doc_id):
        for token in tokens:
            if token not in self.index:
                self.index[token] = [(doc_id, 1)]
            else:
                if self.index[token][-1][0] != doc_id:
                    self.index[token].append((doc_id, 1))
                else:
                    self.index[token][-1] = (doc_id, self.index[token][-1][1] + 1)

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
            f.write('{}\n'.format(self.num_docs))
            for doc, length in self.doc_lengths.items():
                f.write('{}\t{}\n'.format(doc, length))
            for key, postings in self.index.items():
                f.write('{}'.format(key))
                for posting in postings:
                    f.write('\t{}\t{}'.format(posting[0], posting[1]))
                f.write('\n')

    def load_from_file(self, path):
        self.index = {}
        with open(path, encoding='utf-8') as f:
            self.num_docs = int(f.readline().strip())
            for _ in range(self.num_docs):
                line = f.readline().strip()
                parts = line.split('\t')
                self.doc_lengths[int(parts[0])] = int(parts[1])
            while True:
                line = f.readline().strip()
                if line == '':
                    break
                parts = line.split('\t')
                self.index[parts[0]] = [(int(parts[i]), int(parts[i+1])) for i in range(1, len(parts) - 1, 2)]

    def get_postings_list(self, word):
        return self.index[word] if word in self.index else None


# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Stemmer -----------------------------------------------------------

class Stemmer():

    def __init__(self):
        self.suffixes = ['ها', 'ات', 'تر', 'ترین']
        self.verb_prefixes = ['می', 'ن', 'نمی', 'ب']
        self.verb_suffixes = ['م', 'ی', 'د', 'یم', 'ید', 'ند']
        self.erabs = ['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ']
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
        x = r'ء'
        y = r'ئ'
        clean_word = re.sub(x, y, clean_word)
        x = r'ٲ|ٱ|إ|ﺍ|أ'
        y = r'ا'
        clean_word = re.sub(x, y, clean_word)
        x = r'ﺁ|آ'
        y = r'آ'
        clean_word = re.sub(x, y, clean_word)
        x = r'ﺐ|ﺏ|ﺑ'
        y = r'ب'
        clean_word = re.sub(x, y, clean_word)
        x = r'ﭖ|ﭗ|ﭙ|ﺒ|ﭘ'
        y = r'پ'
        clean_word = re.sub(x, y, clean_word)
        x = r'ﭡ|ٺ|ٹ|ﭞ|ٿ|ټ|ﺕ|ﺗ|ﺖ|ﺘ'
        y = r'ت'
        clean_word = re.sub(x, y, clean_word)
        x = r'ﺙ|ﺛ'
        y = r'ث'
        clean_word = re.sub(x, y, clean_word)
        x = r'ﺙ|ﺛ'
        y = r'ث'
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺝ|ڃ|ﺠ|ﺟ"
        y = r"ج"
        clean_word = re.sub(x, y, clean_word)
        x = r"ڃ|ﭽ|ﭼ"
        y = r"چ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺢ|ﺤ|څ|ځ|ﺣ"
        y = r"ح"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺥ|ﺦ|ﺨ|ﺧ"
        y = r"خ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ڏ|ډ|ﺪ|ﺩ"
        y = r"د"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺫ|ﺬ|ﻧ"
        y = r"ذ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ڙ|ڗ|ڒ|ڑ|ڕ|ﺭ|ﺮ"
        y = r"ر"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺰ|ﺯ"
        y = r"ز"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﮊ"
        y = r"ژ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ݭ|ݜ|ﺱ|ﺲ|ښ|ﺴ|ﺳ"
        y = r"س"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺵ|ﺶ|ﺸ|ﺷ"
        y = r"ش"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺺ|ﺼ|ﺻ"
        y = r"ص"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺽ|ﺾ|ﺿ|ﻀ"
        y = r"ض"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻁ|ﻂ|ﻃ|ﻄ"
        y = r"ط"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻆ|ﻇ|ﻈ"
        y = r"ظ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ڠ|ﻉ|ﻊ|ﻋ"
        y = r"ع"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻎ|ۼ|ﻍ|ﻐ|ﻏ"
        y = r"غ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻒ|ﻑ|ﻔ|ﻓ"
        y = r"ف"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻕ|ڤ|ﻖ|ﻗ"
        y = r"ق"
        clean_word = re.sub(x, y, clean_word)
        x = r"ڭ|ﻚ|ﮎ|ﻜ|ﮏ|ګ|ﻛ|ﮑ|ﮐ|ڪ|ك"
        y = r"ک"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﮚ|ﮒ|ﮓ|ﮕ|ﮔ"
        y = r"گ"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻝ|ﻞ|ﻠ|ڵ"
        y = r"ل"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﻡ|ﻤ|ﻢ|ﻣ"
        y = r"م"
        clean_word = re.sub(x, y, clean_word)
        x = r"ڼ|ﻦ|ﻥ|ﻨ"
        y = r"ن"
        clean_word = re.sub(x, y, clean_word)
        x = r"ވ|ﯙ|ۈ|ۋ|ﺆ|ۊ|ۇ|ۏ|ۅ|ۉ|ﻭ|ﻮ|ؤ"
        y = r"و"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﺔ|ﻬ|ھ|ﻩ|ﻫ|ﻪ|ۀ|ە|ة|ہ"
        y = r"ه"
        clean_word = re.sub(x, y, clean_word)
        x = r"ﭛ|ﻯ|ۍ|ﻰ|ﻱ|ﻲ|ں|ﻳ|ﻴ|ﯼ|ې|ﯽ|ﯾ|ﯿ|ێ|ے|ى|ي"
        y = r"ی"
        clean_word = re.sub(x, y, clean_word)
        x = r'¬'
        y = r'‌'
        clean_word = re.sub(x, y, clean_word)
        x = r'•|·|●|·|・|∙|｡|ⴰ'
        y = r'.'
        clean_word = re.sub(x, y, clean_word)
        x = r',|٬|٫|‚|，'
        y = r'،'
        clean_word = re.sub(x, y, clean_word)
        x = r'ʕ'
        y = r'؟'
        clean_word = re.sub(x, y, clean_word)
        x = r'۰|٠'
        y = r'0'
        clean_word = re.sub(x, y, clean_word)
        x = r'۱|١'
        y = r'1'
        clean_word = re.sub(x, y, clean_word)
        x = r'۲|٢'
        y = r'2'
        clean_word = re.sub(x, y, clean_word)
        x = r'۳|٣'
        y = r'3'
        clean_word = re.sub(x, y, clean_word)
        x = r'۴|٤'
        y = r'4'
        clean_word = re.sub(x, y, clean_word)
        x = r'۵'
        y = r'5'
        clean_word = re.sub(x, y, clean_word)
        x = r'۶|٦'
        y = r'6'
        clean_word = re.sub(x, y, clean_word)
        x = r'۷|٧'
        y = r'7'
        clean_word = re.sub(x, y, clean_word)
        x = r'۸|٨'
        y = r'8'
        clean_word = re.sub(x, y, clean_word)
        x = r'۹|٩'
        y = r'9'
        clean_word = re.sub(x, y, clean_word)
        x = r'ـ|ِ|ُ|َ|ٍ|ٌ|ً|'
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
        # s = self.normalize_letters(s)

        # Rule 3: Mokassar plurals
        s = self.plural_to_single(s)

        # Rule 4: Verbs are normalized
        s = self.normalize_if_verb(s)

        # Rule 5: Remove suffix from nouns
        s = self.remove_suffix(s)

        return s

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Tokenizer -----------------------------------------------------------

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

if __name__ == '__main__':
    # test_tokenizer_and_stemmer()
    main()
