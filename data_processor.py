import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_process(file_bytes):
    import io
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        encoding='utf-8-sig',
        on_bad_lines='skip',
        engine='python'
    )

    df.columns = df.columns.str.strip()

    df = df[df['race'] != '?']
    df = df[df['race'].notna()]

    df['target'] = (df['readmitted'] == '<30').astype(int)

    feature_cols = [
        'age', 'gender', 'time_in_hospital',
        'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient',
        'number_diagnoses', 'admission_type_id',
        'admission_source_id', 'discharge_disposition_id',
        'max_glu_serum', 'A1Cresult',
        'metformin', 'insulin', 'diabetesMed'
    ]

    df_model = df[feature_cols + ['race', 'target']].copy()

    # Fill missing values instead of dropping rows
    # For numeric columns use median
    numeric_cols = [
        'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses',
        'admission_type_id', 'admission_source_id',
        'discharge_disposition_id'
    ]
    for col in numeric_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
        df_model[col] = df_model[col].fillna(df_model[col].median())

    # For categorical columns use most common value
    categorical_cols = [
        'age', 'gender', 'max_glu_serum',
        'A1Cresult', 'metformin', 'insulin', 'diabetesMed'
    ]
    for col in categorical_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

    print("Rows after imputation:", len(df_model))

    le = LabelEncoder()
    for col in categorical_cols:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[feature_cols].values
    y = df_model['target'].values
    race = df_model['race'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Final X shape:", X.shape)
    return X, y, race, scaler


def get_race_distribution(race_array):
    unique, counts = np.unique(race_array, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def get_feature_drift(X, race_array):
    from scipy import stats
    races = np.unique(race_array)
    drift_scores = []

    for i in range(X.shape[1]):
        groups = [X[race_array == r, i] for r in races]
        stat, p = stats.kruskal(*groups)
        drift_scores.append(float(stat))

    max_stat = max(drift_scores)
    normalized = [round(s / max_stat, 3) for s in drift_scores]
    return normalized