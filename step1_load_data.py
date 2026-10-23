# step 1: load the synthea csv files into postgres
#
# patients.csv     -> patients      (1 row per person)
# encounters.csv   -> encounters    (1 row per visit)
# conditions.csv   -> conditions    (1 row per diagnosis)
# observations.csv -> observations  (1 row per lab result / vital sign, ~9 million rows)
# medications.csv  -> medications   (1 row per prescription)

import os

import psycopg2

import settings

TABLES = ["patients", "encounters", "conditions", "observations", "medications"]


def main():
    # create empty tables (drops old ones first, so this can be re-run)
    print("Creating tables...")
    settings.run_sql_file("1_create_tables.sql")

    conn = psycopg2.connect(settings.DB_URL)
    cur = conn.cursor()

    for table in TABLES:
        csv_path = os.path.join(settings.CSV_DIR, table + ".csv")
        print("Loading", table)

        # COPY loads the whole file in one go, much faster than inserting row by row
        # HEADER true -> skip the first line with the column names
        with open(csv_path) as f:
            cur.copy_expert("COPY " + table + " FROM STDIN WITH (FORMAT csv, HEADER true)", f)
        conn.commit()

        # row count, to check against wc -l on the csv (csv has 1 extra line for the header)
        cur.execute("SELECT COUNT(*) FROM " + table)
        print("   rows:", cur.fetchone()[0])

    conn.close()


if __name__ == "__main__":
    main()
