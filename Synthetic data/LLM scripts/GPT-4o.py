import pandas as pd
import numpy as np
import random

np.random.seed(42)

# Helper functions
def random_choice_weighted(options, weights):
    return np.random.choice(options, p=weights)

def generate_postcode():
    digits = np.random.randint(1000, 1100)
    letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
    return f"{digits}{letters}"

# Options and weights (rough approximations for Amsterdam population)
ethnicity_opts = ["Nederlands", "Surinaams", "Turks", "Marokkaans", "Anders (specificeer): __________", "Ik wil het liever niet zeggen"]
ethnicity_weights = [0.5, 0.1, 0.1, 0.1, 0.15, 0.05]

religion_opts = ["Geen religie", "Islam", "Christendom", "Hindoeïsme", "Boeddhisme", "Jodendom", "Agnostisch", "Anders (specificeer): __________", "Ik wil het liever niet zeggen"]
religion_weights = [0.4, 0.2, 0.2, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02]

gender_opts = ["Man", "Woman", "Trans man", "Trans woman", "Non-binary", "Other", "Prefer not to say"]
gender_weights = [0.45, 0.45, 0.02, 0.02, 0.03, 0.02, 0.01]

education_opts = ["Geen opleiding", "Basisonderwijs", "Vmbo/Mavo", "Havo/Vwo", "Mbo (niveau 1-4)", "HBO", "Universiteit (Bachelor, Master, of hoger)", "Anders (specificeer): __________"]
education_weights = [0.01, 0.05, 0.15, 0.1, 0.25, 0.25, 0.18, 0.01]

employment_opts = [
    "Betaalde baan (fulltime)", "Gedeeltelijk betaalde baan (parttime)", "Zorg voor het huishouden (eventueel kinderen)",
    "Gepensioneerd of met prepensioen", "Scholier of student, naam opleiding: …", "Vrijwilligerswerk",
    "Gedeeltelijk / geen betaald werk vanwege gezondheidsproblemen", 
    "Geen betaald werk om andere redenen (bv. vanwege onvrijwillige werkloosheid of vrijwilligerswerk)",
    "Anders (specificeer): __________"
]
employment_weights = [0.35, 0.15, 0.05, 0.05, 0.15, 0.05, 0.1, 0.08, 0.02]

# Generate Data
n = 750
data = []
for i in range(n):
    phq = np.random.randint(0, 3, size=9)
    phq_total = sum(phq)
    while not 5 <= phq_total <= 15:
        phq = np.random.randint(0, 4, size=9)
        phq_total = sum(phq)
    TIPI = np.round(np.random.uniform(1.0, 7.0, size=5), 1)
    pos_cr = np.round(np.random.uniform(1.0, 5.0, size=3), 1)
    neg_cr = np.round(np.random.uniform(1.0, 5.0, size=4), 1)
    PANCRS_TotalPositive = round(pos_cr.mean(), 1)
    PANCRS_TotalNegative = round(neg_cr.mean(), 1)
    PANCRS_FrequencyPositive = round(np.random.uniform(1.0, 5.0), 1)
    PANCRS_FrequencyNegative = round(np.random.uniform(1.0, 5.0), 1)
    PANCRS_TotalFrequency = round((PANCRS_FrequencyPositive + PANCRS_FrequencyNegative) / 2, 1)
    row = {
        "PTID": str(i),
        "Age": np.random.randint(18, 70),
        "Sex": random.choice(["Man", "Vrouw"]),
        "CountryOfBirthMother": random.choice(["Netherlands", "Anders (specifeer): Marokko"]),
        "CountryOfBirthFather": random.choice(["Netherlands", "Anders (specifeer): Turkije"]),
        "CountryOfBirth": random.choice(["Netherlands", "Anders (specifeer): Suriname"]),
        "Ethnicity": random_choice_weighted(ethnicity_opts, ethnicity_weights),
        "Nationality": "Nederlands",
        "Religion": random_choice_weighted(religion_opts, religion_weights),
        "Gender": random_choice_weighted(gender_opts, gender_weights),
        "TIPI_Extraversion": TIPI[0],
        "TIPI_Agreeableness": TIPI[1],
        "TIPI_Conscientiousness": TIPI[2],
        "TIPI_Neuroticism": TIPI[3],
        "TIPI_Openness": TIPI[4],
        "EducationLevel": random_choice_weighted(education_opts, education_weights),
        "EmploymentStatus": random_choice_weighted(employment_opts, employment_weights),
        "Postcode": generate_postcode(),
        "PANCRS_Affirmation": pos_cr[0],
        "PANCRS_ProblemSolving": pos_cr[1],
        "PANCRS_EnhancingFriendship": pos_cr[2],
        "PANCRS_TotalPositive": PANCRS_TotalPositive,
        "PANCRS_WorryAboutEvaluation": neg_cr[0],
        "PANCRS_InhibitingHappiness": neg_cr[1],
        "PANCRS_WorryAboutImpact": neg_cr[2],
        "PANCRS_Slack": neg_cr[3],
        "PANCRS_TotalNegative": PANCRS_TotalNegative,
        "PANCRS_FrequencyPositive": PANCRS_FrequencyPositive,
        "PANCRS_FrequencyNegative": PANCRS_FrequencyNegative,
        "PANCRS_TotalFrequency": PANCRS_TotalFrequency,
        "PHQ9_q1": phq[0],
        "PHQ9_q2": phq[1],
        "PHQ9_q3": phq[2],
        "PHQ9_q4": phq[3],
        "PHQ9_q5": phq[4],
        "PHQ9_q6": phq[5],
        "PHQ9_q7": phq[6],
        "PHQ9_q8": phq[7],
        "PHQ9_q9": phq[8],
        "PHQ9_Total": phq_total
    }
    data.append(row)

df = pd.DataFrame(data)
import ace_tools as tools; tools.display_dataframe_to_user(name="WellConnect Cohort Data", dataframe=df)

