import datetime
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from pymorphy2 import MorphAnalyzer
from scipy import sparse
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/usr/lib/chromium-browser/chromedriver')


class Preprocessing():

    def __init__(self, db):
        self.db = db

    def lemmatize(self, txt):
        """
        input:
          txt: str - строка
        output:
          tokens: list - массив преобразованных слов строки

        Обрабатывает поданную на вход строку:
          переводит в нижний регистр
          применяет регулярные выражения для удаления служебных символов
          удаляет все слова не являющиееся существительным/глаголом
          переводит слово в инфинитив
        """

        amounts = ['ед', 'тыс', 'сот', 'млн', 'млрд', 'дес']
        patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-«»|]+"
        stopwords_ru = stopwords.words("russian") + amounts
        morph = MorphAnalyzer()

        txt = txt.lower()
        txt = re.sub(patterns, ' ', txt)
        tokens = []

        for token in txt.split():
            if token and token not in stopwords_ru:
                token = token.strip()
                token = morph.normal_forms(token)[0]

                if token not in amounts and \
                   morph.parse(token)[0].tag.POS in ['NOUN', 'VERB'] and \
                   len(morph.parse(token)[0].normal_form) >= 3:
                    tokens.append(morph.parse(token)[0].normal_form)
        if len(tokens) > 2:
            return tokens
        return None

    def get_lemms(self, df):
        """
        input:
          df - датафрейм с новостями

          колонки:
            header: str - заголовок статьи 
            link: str - ссылка на полный текст статьи
            date: int - время публикации статьи в секундах с 01.01.1970
            topic: str - тематика новости

        Производит лемматизацию переданного датафрейма
        """

        self.lemms = list(map(lambda x: self.lemmatize(x), df.header))
        self.sent = list(map(lambda x: ' '.join(x) if x else '', self.lemms))

    def get_sparse(self):
        """
        output:
          x: sparse matrix - разреженная матрица tf-idf
          words: list - столбцы для x
        """

        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(self.lemms)
        words = list(vectorizer.get_feature_names_out())

        return x, words

    def preprocessing(self, df):
        """
        df - датафрейм с новостями

        колонки:
          header: str - заголовок статьи 
          link: str - ссылка на полный текст статьи
          date: int - время публикации статьи в секундах с 01.01.1970
          topic: str - тематика новости

        Обрабатывает поданный батч данных:
          удаление дубликатов
          обработка топика, удаление всех, не содержащих более двух слов, удаление всех непопулярных топиков
        """

        df.drop_duplicates('header', inplace=True)
        df.topic = df.topic.apply(lambda x: x.lower())
        df = df[(df.topic.isin(df.topic.apply(lambda x: x if len(x.split(' ')) <= 2 else '').unique())) \
                # & (df.topic.isin([key for key, value in Counter(df.topic).items() if value >= 50]))
                ]
        df.sort_values('date', ascending=True, inplace=True)
        df = df.reset_index(drop=True)

        self.get_lemms(df)
        df['preproc_header'] = self.sent
        df = df[df['preproc_header'] != '']

        self.lemms = list(df['preproc_header'].values)
        self.merged = set(' '.join(self.lemms).split(' '))

        sparse, labels = self.get_sparse()

        return sparse, labels, df


class Relevance():

    def __init__(self):
        pass

    @staticmethod
    def func(x, ser):
        '''
        функция для обработки title
        '''
        words = x.split()
        result = 0
        for word in words:
            try:
                result += ser[word]
            except:
                result += 0
        return result

    def get_relevance(self, data, sparse_path, label, is_buisiness):
        '''
        Считываем основной csv, разряженную матрицу с кол-вом встречающихся слов в title, и columns разряженной матрицы.
        После этого выкидываем все слова, которые встречаются меньше 5 раз.
        Находим важность title - сумма частот слов и также время относительно текущей даты.
        Выводи коэффициент важности относительно частоты и времени с коэффициентами 0.1 и 0.9 соответсвенно
        '''
        df = data
        if is_buisiness:
            df = df[df['topic'] != 'финансы']
        else:
            df = df[df['topic'] == 'финансы']

        #df.drop([df.columns.values[0]], axis=1, inplace=True)
        your_mmatrix_back = sparse.load_npz(sparse_path)
        df_m = pd.DataFrame(your_mmatrix_back.toarray())
        df_m[df_m != 0] = 1
        not_zero = [sum(df_m.iloc[:, i] != 0)
                    for i in range(len(df_m.columns))]
        new_cols = [str(col) if qnt > 5 else '' for col,
                    qnt in zip(df_m.columns, not_zero)]
        new_cols = map(lambda x: int(x), ' '.join(new_cols).split())
        df_m = df_m[new_cols]
        with open(label, 'r', encoding='UTF-8') as f:
            result = f.read()
        ser = df_m.sum(axis=0)
        ser.index = np.array(result.split())[df_m.columns.values]

        df = df.dropna()
        df['importance'] = df['preproc_header'].apply(
            lambda x: Relevance.func(x, ser))
        df.importance = df.importance.apply(lambda x: int(x))

        now = datetime.datetime.now()
        dt = datetime.datetime.now()
        s = time.mktime(dt.timetuple())
        df['time'] = df.date.apply(lambda x: (s - float(x)) // (60*60*24))
        df.time[df.time == float(0)] = 1
        df.time = df.time.apply(lambda x: int(x))

        mean_importance = df.importance.mean()
        std_importance = df.importance.std()
        df['importance_standart'] = df.importance.apply(
            lambda x: (x - mean_importance) / std_importance)

        mean_time = df.time.mean()
        std_time = df.time.std()
        df['time_standart'] = df.time.apply(
            lambda x: (x - mean_time) / std_time)

        weight_time = 0.9
        weight_importance = 0.1
        df['weight'] = -weight_time*df.time_standart + \
            weight_importance*df.importance_standart

        df = df.sort_values(by='weight', ascending=False)
        df.index = list(range(1, df.shape[0]+1))
        return df.drop(['preproc_header', 'importance', 'time', 'importance_standart', 'time_standart'], axis=1)
