# step 2: build the modelling table in postgres
#
# 2_cohort.sql   -> who is in the study + label (diagnosed in next 12 months yes/no)
# 3_features.sql -> features from the 2 years before each snapshot
#
# result: table "features", 1 row per patient per yearly snapshot

import settings


def main():
    print("Building cohort...")
    settings.run_sql_file("2_cohort.sql")

    print("Building features...")
    settings.run_sql_file("3_features.sql")

    # check: patients and positives per snapshot (should be around 0.5% positives)
    summary = settings.query_to_df("""
        SELECT index_date,
               COUNT(*)   AS patients,
               SUM(label) AS new_kidney_patients
        FROM features
        GROUP BY index_date
        ORDER BY index_date
    """)
    summary["percent"] = (100 * summary["new_kidney_patients"] / summary["patients"]).round(2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
