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
        self.suffixes = ['ها' ,'ات' ,'تر' ,'ترین' ,'آسا' ,'اسا' ,'سان']
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
                if (temp in self.bon_mazi) or (temp in self.bon_mozare):
                    return temp
            if verb.endswith(suf):
                temp = verb[:-len(suf)]
                if (temp in self.bon_mazi) or (temp in self.bon_mozare):
                    return temp
            if verb.startswith(pre) and verb.endswith(suf):
                temp = verb[len(pre):-len(suf)]
                if (temp in self.bon_mazi) or (temp in self.bon_mozare):
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

        # Rule 1: Remove suffix from nouns
        # s = self.remove_suffix(s) 

        # Rule 2: Normalize letters in words
        # s = self.normalize_letters(s)

        # Rule 3: Verbs are normalized
        # s = self.normalize_if_verb(s)

        # Rule 4: Mokassar plurals
        # s = self.plural_to_single(s)

        # Rule 5: Remove erabs
        # s = self.remove_erabs(s)

        return s


if __name__ == '__main__':
    
    test_stem = Stemmer()
    new = test_stem.stem('محمّد')
    print(new)
