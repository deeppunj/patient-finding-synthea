# shared settings + db helpers, imported by all the step files

import os

import pandas as pd
import psycopg2

# project folder (so paths work no matter where the script is started from)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# synthea writes the csv files to output/csv
CSV_DIR = os.path.join(PROJECT_DIR, "output", "csv")
SQL_DIR = os.path.join(PROJECT_DIR, "sql")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

# postgres connection
# format: postgresql://user:password@host:port/dbname
# if your db needs a password, set PF_DB_URL as env variable instead of writing it here
# (keeps the password out of github)
DB_URL = os.environ.get("PF_DB_URL", "postgresql://localhost:5432/patientfinding")

# mlflow stores the runs in a sqlite file in the project folder
MLFLOW_URI = "sqlite:///" + os.path.join(PROJECT_DIR, "mlflow.db")

# snapshots before this date = training, from this date = test
TEST_FROM = "2024-01-01"

# end of the synthea data (the -e 20261001 in the readme). nothing is known after this day,
# so step 5 (survival analysis) stops following the test patients here
DATA_END = "2026-10-01"


def run_sql_file(file_name):
    # read the .sql file and run it on the database
    with open(os.path.join(SQL_DIR, file_name)) as f:
        sql = f.read()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()  # without commit the new tables are not saved
    conn.close()


def query_to_df(query):
    # run a select query, return the result as a dataframe
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(query)

    # column names are in cur.description
    columns = [col[0] for col in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=columns)

    conn.close()
    return df
