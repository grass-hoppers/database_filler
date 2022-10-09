

import numpy as np
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import warnings
import numpy as np
import tensorflow as tf
warnings.filterwarnings("ignore")
nltk.download("stopwords")

class Duplicates(object):
    def __init__(self, data):
        self.data = data
        self.titles = data['header']
        self.times = data['date']

        self.tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

    # Нужно подключение к интернету для загрузки модели
    def embed_bert_cls(self, text, model, tokenizer):
        with tf.device('/GPU:0'):
            t = tokenizer(text, padding=True, truncation=True,
                          return_tensors='pt')
            with torch.no_grad():
                model_output = model(**{k: v.to(model.device)
                                     for k, v in t.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)
            return embeddings[0].cpu().numpy()

    def get_unique_data(self):
        #simular_titles_to_delete = duplicate_indexes
        simular_titles_to_delete = []
        eps = 0.05
        one_day_time = 60 * 60 * 24

        count_news = len(self.titles)

        matrix_simularity = np.zeros((count_news, count_news))
        # Матрица эмбедингов всех заголовков
        emb_titles2 = torch.zeros((count_news, 312))
        # Каждый эмбединг заголовка будет сравниваться со всеми остальными эмбедингами заголовков
        for i in range(count_news):
            if i not in simular_titles_to_delete:
                # Вектор эмбединга проверяемого заголовка
                emb_titles1 = torch.zeros((count_news, 312))
                if self.titles[i] != None:
                    # Вычисляется эмбединг проверяемого заголовка
                    emb_titles1 = torch.tensor(self.embed_bert_cls(
                        self.titles[i], self.model, self.tokenizer))
                if i == 0:
                    # Вычисляются эмбединги всех заголовков и записываются в матрицу
                    for j in range(count_news):
                        # if self.titles[j] != None:
                        emb_titles2[j] = torch.tensor(self.embed_bert_cls(
                            self.titles[j], self.model, self.tokenizer))
                        if j % 1000 == 0:
                            print('got embedding of', j, 'titles')
                if i % 100 == 0:
                    print('cheked', i, 'articles on duplicates')
                # Сохраняется вектор сравнения рассматриваемого заголовка со всеми остальными
                matrix_simularity[i] = torch.nn.functional.cosine_similarity(
                    torch.tensor(emb_titles1), torch.tensor(emb_titles2)).cpu().detach().numpy()
                # Обнуляется значения сравнения эмбединга заголовка с собой же
                matrix_simularity[i, i] = 0
                max_simul_arg = matrix_simularity[i, i:].argmax() + i
                # Если выбранные эмбеддинги заголовков удовлетворяют условиям, то они запоминаются как дубликаты
                try:
                    if matrix_simularity[i, max_simul_arg] > 0.8 and abs(int(self.times[i]) - int(self.times[max_simul_arg])) < one_day_time * 3:
                        simular_titles_to_delete.append(max_simul_arg)
                except Exception as e:
                    print('Ошибка ' + 'i= ' + str(i) +
                          ' max_simul_index= ' + str(max_simul_arg))

        uniq_titles = []     # Список индексов уникальных статей
        for i in range(count_news):
            if i not in simular_titles_to_delete:
                uniq_titles.append(i)
        # print(uniq_titles)
        return self.data.iloc[uniq_titles, :]
