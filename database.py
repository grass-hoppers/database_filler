

import psycopg2
import warnings
import time
warnings.filterwarnings("ignore")



class DB():

    def __init__(self, host, port, dbname, user, password):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password

        self.conn = psycopg2.connect(
            host=self.host,
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            password=self.password
        )

    def commit(self):
        self.conn.commit()

    def insrt(self, row, name_table):
        try:
            names = ', '.join(list(row.index))
            vals = ', '.join(
                list(map(lambda x: f"'{x}'" if type(x) in [str] else str(x), list(row))))
            # return f"""INSERT INTO {name_table} ({names}) VALUES ({vals})"""
            crs = self.conn.cursor()
            crs.execute(
                f"""INSERT INTO {name_table} ({names}) VALUES ({vals})""")
            time.sleep(0.1)
        except (Exception) as error:
            print("Ошибка при работе с PostgreSQL", error)
            crs.close()

    def get_all(self, table):
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password)

            crs = conn.cursor()
            crs.execute(f"""SELECT * FROM {table}""")
            df = crs.fetchall()

            return df
        except (Exception) as error:
            print("Ошибка при работе с PostgreSQL", error)
        finally:
            if conn:
                crs.close()
                conn.close()
                print("Соединение с PostgreSQL закрыто")
