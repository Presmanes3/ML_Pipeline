import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Config ---
DATA_PATH = "./data/processed/NY-House-Dataset-Cleaned.csv"  # dataset ya limpio
OUTPUT_DIR = "./data/processed"                              # donde guardar train/test
TEST_SIZE = 0.2
RANDOM_STATE = 21


def main():
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("✂️ Splitting into train and test...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train set saved at: {train_path} ({len(train_df)} rows)")
    print(f"✅ Test set saved at: {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
