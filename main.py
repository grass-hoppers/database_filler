from database import DB
from duplicates_checker import Duplicates
from parsers import Parser
from preprocessing import Preprocessing, Relevance
from frequency import FrequencyAnalysis

import pandas as pd
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")


def get_new_news(prs, n_iters):
    prs.infinite_scroll(ScrollNumber=n_iters)
    prs.button_scroll(ScrollNumber=n_iters)
    prs.parse_walking(pages_to_parse=n_iters)


def update_news_preprocessing(db):
    df = pd.DataFrame(db.get_all(table='news'), columns=[
        'ind', 'header', 'link', 'date', 'topic']).drop('ind', axis=1)
    p = Preprocessing(db=db)
    sprs, labels, df = p.preprocessing(df)
    df = df.reset_index(drop=True)
    df = Duplicates(df).get_unique_data()
    new_ind = list(df.index)
    sprs = sparse.csr_matrix(pd.DataFrame(
        sprs.todense()).iloc[new_ind, :].values)

    for ind, row in df.iterrows():
        db.insrt(row=row, name_table='news_preprocessing')
    db.commit()

    sparse_df = pd.DataFrame(sprs.todense())
    sparse.save_npz(f'sparse_.npz', sprs)
    with open(f'labels_business.txt', 'w') as f:
        f.write(' '.join(list(labels)))


def update_relevance(db):
    df = pd.DataFrame(db.get_all(table='news_preprocessing'), columns=[
        'ind', 'header', 'link', 'date', 'topic', 'preproc_header']).drop('ind', axis=1)
    print(df.columns)
    df_ = Relevance().get_relevance(df, '/content/sparse_.npz',
                                    f'/content/labels.txt', True).rename(columns={'weight': 'importance'})
    print(df_.columns)
    for ind, row in df_.iterrows():
        db.insrt(row=row, name_table='news_relevance')
    db.commit()


if __name__ == "__main__":

    PATH = ''

    db = DB(host='rc1b-h8wfencozid6bilo.mdb.yandexcloud.net',
            port=6432,
            user='user1',
            dbname='db1',
            password='5XaqgNhk^8t76zPV')
    prs = Parser(db)

    get_new_news(prs, 2)
    update_news_preprocessing(db)
    db.commit()
    update_relevance(db)
