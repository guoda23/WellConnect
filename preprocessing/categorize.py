import pandas as pd
import numpy as np

# Age

def categorize_age_binary(df, cut=35, new_col="Age_binary"):
    """
    Add a binary age column with string labels:
    '<cut' or '≥cut', e.g. '<35' or '≥35'.
    """
    df[new_col] = np.where(df["Age"] >= cut,
                           f"≥{cut}",
                           f"<{cut}")
    return df


def categorize_age_tertiary(df, cuts=(-np.inf, 30, 50, np.inf),
                   labels=("young","mid","older"),
                   new_col="Age_tertiary"):
    """
    Add a 3-category age column based on cutpoints.
    """
    df[new_col] = pd.cut(df["Age"], bins=cuts, labels=labels)
    return df

# Education

def categorize_education_binary(df, col="EducationLevel", new_col="EducationLevel_binary", random_state=None):
    mapping = {
        "Geen opleiding": "<HBO",
        "Basisonderwijs": "<HBO",
        "Vmbo/Mavo": "<HBO",
        "Mbo (niveau 1-4)": "<HBO",
        "Havo/Vwo": "<HBO",
        "HBO": "HBO/WO",
        "Universiteit (Bachelor, Master, of hoger)": "HBO/WO",
    }
    df[new_col] = df[col].map(mapping)

    # Students → probabilistic assignment
    mask = df[col] == "Scholier of student, naam opleiding: …"
    rng = np.random.default_rng(random_state)  # seeded RNG
    df.loc[mask, new_col] = rng.choice(
        ["<HBO", "HBO/WO"],
        size=mask.sum(),
        p=[0.63, 0.37]
    )

    # Flag unmapped values
    unmapped = df.loc[df[new_col].isna(), col].unique()
    if len(unmapped) > 0:
        print(f"[Warning] Unmapped values in {col}: {unmapped}")

    df[new_col] = df[new_col].fillna("Unknown")
    return df


def categorize_education_tertiary(df, col="EducationLevel", new_col="EducationLevel_tertiary", random_state=None):
    mapping = {
        "Geen opleiding": "Low",
        "Basisonderwijs": "Low",
        "Vmbo/Mavo": "Low",
        "Mbo (niveau 1-4)": "Medium",
        "Havo/Vwo": "Medium",
        "HBO": "High",
        "Universiteit (Bachelor, Master, of hoger)": "High",
    }
    df[new_col] = df[col].map(mapping)

    # Probabilistic assignment for students
    student_mask = df[col] == "Scholier of student, naam opleiding: …"
    probs = np.array([0.26, 0.37, 0.364], dtype=float)
    probs = probs / probs.sum()
    choices = ["Low", "Medium", "High"]

    rng = np.random.default_rng(random_state)  # seeded RNG
    df.loc[student_mask, new_col] = rng.choice(
        choices, size=student_mask.sum(), p=probs
    )

    # Flag unmapped values
    unmapped = df.loc[df[new_col].isna(), col].unique()
    if len(unmapped) > 0:
        print(f"[Warning] Unmapped values in {col}: {unmapped}")
        df[new_col] = df[new_col].fillna("Unknown")

    return df


# Gender

def categorize_gender_binary(df, col="Sex", new_col="Gender_binary"):
    """Binary gender column is equivalent to the sex column"""
    df[new_col] = df[col]
    return df

def categorize_gender_tertiary(df, col="Gender", new_col="Gender_tertiary"):
    mapping = {
        "Man": "Male",
        "Vrouw": "Female"
    }
    df[new_col] = df[col].apply(lambda x: mapping.get(x, "Other"))
    return df
