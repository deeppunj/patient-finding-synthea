# runs step 1 to 5 in order
#
# python run_all.py              -> everything
# python run_all.py --skip-load  -> skip step 1 (data only needs to be loaded once)
#
# in pycharm: add --skip-load under Run > Edit Configurations > Script parameters

import sys

import step1_load_data
import step2_build_dataset
import step3_train_model
import step4_insights
import step5_survival

if "--skip-load" not in sys.argv:
    print("=== Step 1: load data ===")
    step1_load_data.main()

print("\n=== Step 2: build dataset ===")
step2_build_dataset.main()

print("\n=== Step 3: train models ===")
step3_train_model.main()

print("\n=== Step 4: insights ===")
step4_insights.main()

print("\n=== Step 5: survival analysis ===")
step5_survival.main()

print("\nDone. Next: mlflow ui --backend-store-uri sqlite:///mlflow.db   and   streamlit run dashboard/app.py")
