import pandas as pd

import numpy as np

import random

import uuid

 

 

def generate_phq9_scores():

    """

    Generates correlated PHQ-9 scores, ensuring the total score

    is mostly in the mild-to-moderate range (5-15).

    """

    # Generate total score with a distribution centered on mild/moderate

    total_score = int(

        np.clip(np.random.normal(loc=11, scale=3.5), 2, 22)

    )

 

    # Distribute the total score among the 9 questions

    questions = np.zeros(9, dtype=int)

    for _ in range(total_score):

        # Find questions that can still be incremented

        eligible_indices = np.where(questions < 3)[0]

        if len(eligible_indices) == 0:

            # This can happen if the total score is > 27, but we clipped it.

            # Or if distribution is unlucky. Re-balance if needed.

            break

        # Pick one eligible question at random and increment it

        choice = random.choice(eligible_indices)

        questions[choice] += 1

 

    # Ensure the sum is exactly the generated total

    while questions.sum() < total_score:

        eligible_indices = np.where(questions < 3)[0]

        if len(eligible_indices) == 0:

            break

        choice = random.choice(eligible_indices)

        questions[choice] += 1

    # In the rare case of overshoot, decrement

    while questions.sum() > total_score:

        eligible_indices = np.where(questions > 0)[0]

        if len(eligible_indices) == 0:

            break

        choice = random.choice(eligible_indices)

        questions[choice] -= 1

 

    phq9_data = {f"PHQ9_q{i+1}": questions[i] for i in range(9)}

    phq9_data["PHQ9_Total"] = questions.sum()

    return phq9_data

 

 

def generate_correlated_scores(phq9_total):

    """

    Generates TIPI and PANCRS scores correlated with depression severity.

    """

    # Normalize PHQ-9 score (0 to 1) for correlation base

    phq9_norm = phq9_total / 27.0

 

    # TIPI Scores (1.0-7.0)

    # Neuroticism positively correlated with PHQ-9

    neuro_base = 1.5 + phq9_norm * 5.0

    neuroticism = np.clip(

        np.random.normal(loc=neuro_base, scale=0.8), 1.0, 7.0

    )

    # Extraversion negatively correlated with PHQ-9

    extra_base = 6.0 - phq9_norm * 4.0

    extraversion = np.clip(

        np.random.normal(loc=extra_base, scale=1.0), 1.0, 7.0

    )

    # Other TIPI scores are more random

    agreeableness = np.clip(np.random.normal(loc=4.8, scale=1.0), 1.0, 7.0)

    conscientiousness = np.clip(

        np.random.normal(loc=5.0, scale=1.2), 1.0, 7.0

    )

    openness = np.clip(np.random.normal(loc=5.2, scale=1.0), 1.0, 7.0)

 

    # PANCRS Scores (1.0-5.0)

    # Negative co-rumination correlated with neuroticism/PHQ-9

    neg_base = 1.5 + (neuroticism / 7.0) * 3.0

    worry_eval = np.clip(

        np.random.normal(loc=neg_base, scale=0.6), 1.0, 5.0

    )

    inhibit_happy = np.clip(

        np.random.normal(loc=neg_base, scale=0.6), 1.0, 5.0

    )

    worry_impact = np.clip(

        np.random.normal(loc=neg_base, scale=0.6), 1.0, 5.0

    )

    slack = np.clip(np.random.normal(loc=neg_base - 0.5, scale=0.7), 1.0, 5.0)

    pancrs_total_negative = np.mean(

        [worry_eval, inhibit_happy, worry_impact, slack]

    )

 

    # Positive co-rumination less correlated

    pos_base = 4.5 - (neuroticism / 7.0) * 2.0

    affirmation = np.clip(

        np.random.normal(loc=pos_base, scale=0.6), 1.0, 5.0

    )

    problem_solving = np.clip(

        np.random.normal(loc=pos_base - 0.4, scale=0.6), 1.0, 5.0

    )

    enhancing_friendship = np.clip(

        np.random.normal(loc=pos_base, scale=0.6), 1.0, 5.0

    )

    pancrs_total_positive = np.mean(

        [affirmation, problem_solving, enhancing_friendship]

    )

 

    # Frequency scores

    freq_pos = np.clip(

        pancrs_total_positive - np.random.normal(0, 0.3), 1.0, 5.0

    )

    freq_neg = np.clip(

        pancrs_total_negative - np.random.normal(0, 0.3), 1.0, 5.0

    )

    pancrs_total_freq = np.mean([freq_pos, freq_neg])

 

    return {

        "TIPI_Extraversion": round(extraversion, 1),

        "TIPI_Agreeableness": round(agreeableness, 1),

        "TIPI_Conscientiousness": round(conscientiousness, 1),

        "TIPI_Neuroticism": round(neuroticism, 1),

        "TIPI_Openness": round(openness, 1),

        "PANCRS_Affirmation": round(affirmation, 1),

        "PANCRS_ProblemSolving": round(problem_solving, 1),

        "PANCRS_EnhancingFriendship": round(enhancing_friendship, 1),

        "PANCRS_TotalPositive": round(pancrs_total_positive, 1),

        "PANCRS_WorryAboutEvaluation": round(worry_eval, 1),

        "PANCRS_InhibitingHappiness": round(inhibit_happy, 1),

        "PANCRS_WorryAboutImpact": round(worry_impact, 1),

        "PANCRS_Slack": round(slack, 1),

        "PANCRS_TotalNegative": round(pancrs_total_negative, 1),

        "PANCRS_FrequencyPositive": round(freq_pos, 1),

        "PANCRS_FrequencyNegative": round(freq_neg, 1),

        "PANCRS_TotalFrequency": round(pancrs_total_freq, 1),

    }

 

 

def generate_demographics():

    """

    Generates realistic demographic data for a participant in Amsterdam.

    """

    # Age distribution

    age = int(

        np.clip(

            np.random.normal(loc=38, scale=15), 18, 85

        )  # Centered around late 30s

    )

 

    # Sex and Gender

    sex = random.choices(["Vrouw", "Man"], weights=[0.52, 0.48], k=1)[0]

    if random.random() < 0.98:

        gender = sex

    else:

        gender = random.choice(

            ["Trans man", "Trans woman", "Non-binary", "Other"]

        )

 

    # Ethnicity and Country of Birth (based on Amsterdam stats)

    ethnicities = [

        "Nederlands",

        "Marokkaans",

        "Surinaams",

        "Turks",

        "Anders (specificeer): __________",

        "Ik wil het liever niet zeggen",

    ]

    # Approximate weights for Amsterdam

    weights = [0.52, 0.11, 0.09, 0.06, 0.18, 0.04]

    ethnicity = random.choices(ethnicities, weights=weights, k=1)[0]

 

    country_of_birth_mother = "Nederland"

    country_of_birth_father = "Nederland"

    country_of_birth = "Nederland"

    nationality = "Nederlands"

 

    if ethnicity == "Marokkaans":

        if random.random() < 0.6:  # First generation

            country_of_birth = "Marokko"

            nationality = "Marokkaans"

        country_of_birth_mother = "Marokko"

        country_of_birth_father = "Marokko"

    elif ethnicity == "Turks":

        if random.random() < 0.55:

            country_of_birth = "Turkije"

            nationality = "Turks"

        country_of_birth_mother = "Turkije"

        country_of_birth_father = "Turkije"

    elif ethnicity == "Surinaams":

        if random.random() < 0.4:

            country_of_birth = "Suriname"

        country_of_birth_mother = "Suriname"

        country_of_birth_father = "Suriname"

    elif "Anders" in ethnicity:

        other_countries = [

            "Duitsland",

            "Verenigd Koninkrijk",

            "Polen",

            "Indonesië",

            "Ghana",

            "Brazilië",

            "Verenigde Staten",

        ]

        chosen_country = random.choice(other_countries)

        ethnicity = f"Anders (specificeer): {chosen_country}"

        if random.random() < 0.8:  # Assume most are immigrants

            country_of_birth = chosen_country

            nationality = f"{chosen_country}"

            country_of_birth_mother = chosen_country

            country_of_birth_father = chosen_country

        else:  # Or second gen

            country_of_birth_mother = chosen_country

            country_of_birth_father = chosen_country

 

    # Education Level

    edu_levels = [

        "Geen opleiding",

        "Basisonderwijs",

        "Vmbo/Mavo",

        "Havo/Vwo",

        "Mbo (niveau 1-4)",

        "HBO",

        "Universiteit (Bachelor, Master, of hoger)",

    ]

    edu_weights = [0.02, 0.05, 0.15, 0.18, 0.25, 0.20, 0.15]

    education = random.choices(edu_levels, weights=edu_weights, k=1)[0]

 

    # Employment Status (correlated with age)

    emp_statuses = [

        "Betaalde baan (fulltime)",

        "Gedeeltelijk betaalde baan (parttime)",

        "Zorg voor het huishouden (eventueel kinderen)",

        "Gepensioneerd of met prepensioen",

        "Scholier of student, naam opleiding: …",

        "Vrijwilligerswerk",

        "Gedeeltelijk / geen betaald werk vanwege gezondheidsproblemen",

        "Geen betaald werk om andere redenen (bv. vanwege onvrijwillige werkloosheid of vrijwilligerswerk)",

    ]

    if age < 25:

        emp_weights = [0.15, 0.25, 0.02, 0.0, 0.45, 0.03, 0.05, 0.05]

    elif age > 65:

        emp_weights = [0.01, 0.05, 0.10, 0.70, 0.0, 0.10, 0.03, 0.01]

    else:

        emp_weights = [0.40, 0.25, 0.08, 0.0, 0.02, 0.05, 0.12, 0.08]

    employment = random.choices(emp_statuses, weights=emp_weights, k=1)[0]

 

    # Religion

    religions = [

        "Geen religie",

        "Christendom",

        "Islam",

        "Agnostisch",

        "Boeddhisme",

        "Hindoeïsme",

        "Ik wil het liever niet zeggen",

    ]

    rel_weights = [0.58, 0.15, 0.12, 0.05, 0.04, 0.03, 0.03]

    religion = random.choices(religions, weights=rel_weights, k=1)[0]

 

    # Postcode

    amsterdam_prefixes = [

        "1011",

        "1012",

        "1013",

        "1014",

        "1015",

        "1016",

        "1017",

        "1018",

        "1019",

        "1021",

        "1022",

        "1023",

        "1024",

        "1025",

        "1031",

        "1033",

        "1034",

        "1035",

        "1051",

        "1052",

        "1053",

        "1054",

        "1055",

        "1056",

        "1057",

        "1058",

        "1059",

        "1061",

        "1062",

        "1063",

        "1064",

        "1065",

        "1066",

        "1067",

        "1068",

        "1069",

        "1071",

        "1072",

        "1073",

        "1074",

        "1075",

        "1076",

        "1077",

        "1078",

        "1081",

        "1082",

        "1083",

        "1086",

        "1087",

        "1091",

        "1092",

        "1093",

        "1094",

        "1095",

        "1096",

        "1097",

        "1098",

        "1101",

        "1102",

        "1103",

        "1104",

        "1105",

        "1106",

        "1107",

    ]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    postcode = (

        random.choice(amsterdam_prefixes)

        + random.choice(letters)

        + random.choice(letters)

    )

 

    return {

        "Age": age,

        "Sex": sex,

        "CountryOfBirthMother": country_of_birth_mother,

        "CountryOfBirthFather": country_of_birth_father,

        "CountryOfBirth": country_of_birth,

        "Ethnicity": ethnicity,

        "Nationality": nationality,

        "Religion": religion,

        "Gender": gender,

        "EducationLevel": education,

        "EmploymentStatus": employment,

        "Postcode": postcode,

    }

 

 

def generate_dataset(num_rows=750):

    """

    Main function to generate the full dataset.

    """

    patient_data = []

    for i in range(num_rows):

        # Generate a unique patient ID

        ptid = f"WC{i+1:04d}"

 

        # Generate data for one patient

        demographics = generate_demographics()

        phq9_scores = generate_phq9_scores()

        correlated_scores = generate_correlated_scores(

            phq9_scores["PHQ9_Total"]

        )

 

        # Combine all data into a single record

        record = {

            "PTID": ptid,

            **demographics,

            **correlated_scores,

            **phq9_scores,

        }

        patient_data.append(record)

 

    # Create DataFrame

    df = pd.DataFrame(patient_data)

 

    # Ensure correct column order

    column_order = [

        "PTID",

        "Age",

        "Sex",

        "CountryOfBirthMother",

        "CountryOfBirthFather",

        "CountryOfBirth",

        "Ethnicity",

        "Nationality",

        "Religion",

        "Gender",

        "TIPI_Extraversion",

        "TIPI_Agreeableness",

        "TIPI_Conscientiousness",

        "TIPI_Neuroticism",

        "TIPI_Openness",

        "EducationLevel",

        "EmploymentStatus",

        "Postcode",

        "PANCRS_Affirmation",

        "PANCRS_ProblemSolving",

        "PANCRS_EnhancingFriendship",

        "PANCRS_TotalPositive",

        "PANCRS_WorryAboutEvaluation",

        "PANCRS_InhibitingHappiness",

        "PANCRS_WorryAboutImpact",

        "PANCRS_Slack",

        "PANCRS_TotalNegative",

        "PANCRS_FrequencyPositive",

        "PANCRS_FrequencyNegative",

        "PANCRS_TotalFrequency",

        "PHQ9_q1",

        "PHQ9_q2",

        "PHQ9_q3",

        "PHQ9_q4",

        "PHQ9_q5",

        "PHQ9_q6",

        "PHQ9_q7",

        "PHQ9_q8",

        "PHQ9_q9",

        "PHQ9_Total",

    ]

    df = df[column_order]

 

    return df

 

 

if __name__ == "__main__":

    # Generate the dataset with 750 rows

    wellconnect_df = generate_dataset(num_rows=750)

 

    # To display the CSV output directly:

    # print(wellconnect_df.to_csv(index=False, quoting=1))

 

    # To save to a file:

    # wellconnect_df.to_csv("wellconnect_cohort_data.csv", index=False, quoting=1)

 

    print("--- Python Script for Data Generation ---")

    with open(__file__, "r") as f:

        print(f.read())

    print("\n--- Sample of Generated Data (first 5 rows) ---")

    print(wellconnect_df.head().to_string())

    print(f"\nGenerated a DataFrame with {len(wellconnect_df)} rows.")