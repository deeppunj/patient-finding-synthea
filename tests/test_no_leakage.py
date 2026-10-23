# checks on the dataset from step 2, run with: pytest
# main thing: no data from on/after the index date may end up in the features

import settings


def count(query):
    # for queries that return 1 number
    return settings.query_to_df(query).iloc[0, 0]


def test_nobody_already_has_kidney_disease():
    # people diagnosed before the index date should not be in the cohort
    assert count("""
        SELECT COUNT(*) FROM cohort c
        JOIN first_kidney_dx k ON k.patient_id = c.patient_id
        WHERE k.first_dx_date < c.index_date
    """) == 0


def test_labels_match_the_12_month_window():
    # recompute the label and count rows where it's different
    assert count("""
        SELECT COUNT(*) FROM cohort c
        LEFT JOIN first_kidney_dx k ON k.patient_id = c.patient_id
        WHERE c.label <> CASE WHEN k.first_dx_date >= c.index_date
                               AND k.first_dx_date < c.index_date + INTERVAL '1 year'
                              THEN 1 ELSE 0 END
    """) == 0


def test_features_use_only_past_data():
    # last visit and last lab used must be before the index date
    assert count("""
        SELECT COUNT(*) FROM features
        WHERE last_visit_date >= index_date OR last_lab_date >= index_date
    """) == 0
