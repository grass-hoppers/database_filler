from database import DB
from duplicates_checker import Duplicates
from parsers import Parser
from preprocessing import Preprocessing, Relevance
from frequency import FrequencyAnalysis

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    PATH = ''
    N = 5
    db = DB(host='rc1b-h8wfencozid6bilo.mdb.yandexcloud.net',
            port=6432,
            user='user1',
            dbname='db1',
            password='5XaqgNhk^8t76zPV')
    db.conn.cursor().execute('DELETE FROM trends')
    cntrs = FrequencyAnalysis(db,3).get_counters()
    trends = []
    freqs = []
    for cnt in cntrs:
        for ngrm, frq in cnt.most_common(5)[2:]:
            trends.append(' '.join(ngrm))
            freqs.append(frq)
    df = pd.DataFrame([trends, freqs], columns=['trend', 'freaquency'])

    for ind, row in df.iterrows():
        db.insrt(row, trends)
    
