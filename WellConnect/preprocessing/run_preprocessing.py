#!/usr/bin/env python3
import argparse
import pandas as pd

from categorize import (
    categorize_age_binary,
    categorize_age_tertiary,
    categorize_education_binary,
    categorize_education_tertiary,
    categorize_gender_binary,
    categorize_gender_tertiary,
)

def main(input_file, output_file, random_state=None):
    df = pd.read_csv(input_file)

    # Apply categorizations
    df = categorize_age_binary(df)
    df = categorize_age_tertiary(df)
    df = categorize_education_binary(df, random_state=random_state)
    df = categorize_education_tertiary(df, random_state=random_state)
    df = categorize_gender_binary(df)
    df = categorize_gender_tertiary(df)

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing on CSV data.")
    parser.add_argument("input_file", help="Path to input CSV")
    parser.add_argument("output_file", help="Path to save processed CSV")
    parser.add_argument("--random_state", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.random_state)
