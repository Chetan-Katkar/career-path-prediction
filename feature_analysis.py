import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PART 1: Loading and Encoding Data")
print("=" * 60)

df = pd.read_csv(r'd:\internship\PS2_Dataset.csv')

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

target_col = 'Suggested Job Role'
y_raw = df[target_col]

target_enc = LabelEncoder()
y = target_enc.fit_transform(y_raw)
print(f"Target classes: {list(target_enc.classes_)}")
print(f"Total samples: {len(df)}")

print("\n" + "=" * 60)
print("PART 2: Feature-Target Correlation Analysis")
print("=" * 60)

df_encoded = df.drop(target_col, axis=1).copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

print("\n--- Mutual Information Scores ---")
print("(Higher = stronger relationship with target)")
print("(Score near 0 = feature tells us NOTHING about the job role)\n")

mi_scores = mutual_info_classif(df_encoded, y, random_state=42)
mi_df = pd.DataFrame({
    'Feature': df_encoded.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

for _, row in mi_df.iterrows():
    bar = '#' * int(row['MI_Score'] * 100)
    print(f"  {row['Feature']:45s} {row['MI_Score']:.4f}  {bar}")

print("\n--- Chi-Squared Scores ---")
print("(Higher = more statistically significant relationship)\n")

chi2_scores, p_values = chi2(df_encoded, y)
chi2_df = pd.DataFrame({
    'Feature': df_encoded.columns,
    'Chi2_Score': chi2_scores,
    'P_Value': p_values
}).sort_values('Chi2_Score', ascending=False)

for _, row in chi2_df.iterrows():
    sig = "SIGNIFICANT" if row['P_Value'] < 0.05 else "NOT significant"
    print(f"  {row['Feature']:45s} Chi2={row['Chi2_Score']:8.2f}  p={row['P_Value']:.4f}  [{sig}]")

print("\n--- Feature Value Distribution ---")
print("(If all values are nearly equally distributed, feature is likely random)\n")
for col in df.drop(target_col, axis=1).columns:
    unique_vals = df[col].nunique()
    print(f"  {col:45s} {unique_vals:3d} unique values")

print("\n" + "=" * 60)
print("PART 3: Cross-Tabulation Check")
print("=" * 60)
print("(If a feature predicts the target, certain values should")
print(" appear much more often for specific job roles)\n")

top_features = mi_df.head(3)['Feature'].tolist()
for feat in top_features:
    print(f"\n--- {feat} vs {target_col} ---")
    ct = pd.crosstab(df[feat], df[target_col], normalize='index')
    max_val = ct.max().max()
    min_val = ct.min().min()
    print(f"  Expected if random: {100/12:.1f}% per cell")
    print(f"  Actual range: {min_val*100:.1f}% to {max_val*100:.1f}%")
    if max_val < 0.15:
        print(f"  --> VERDICT: Very weak relationship (nearly random)")
    else:
        print(f"  --> VERDICT: Some pattern detected!")

print("\n" + "=" * 60)
print("PART 4: Feature Engineering Experiments")
print("=" * 60)

X_base = df_encoded.copy()

print("\n--- Experiment 1: Baseline (Label Encoding) ---")
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
xgb = XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
xgb.fit(X_train, y_train)
acc1 = accuracy_score(y_test, xgb.predict(X_test))
print(f"  XGBoost Accuracy: {acc1*100:.2f}%")

print("\n--- Experiment 2: Adding Engineered Features ---")
X_eng = X_base.copy()
X_eng['technical_score'] = X_eng['coding skills rating'] + X_eng['Logical quotient rating'] + X_eng['hackathons']
X_eng['communication_score'] = X_eng['public speaking points'] + X_eng['reading and writing skills']
X_eng['code_vs_speak'] = X_eng['coding skills rating'] / (X_eng['public speaking points'] + 1)
X_eng['total_score'] = X_eng['Logical quotient rating'] + X_eng['hackathons'] + X_eng['coding skills rating'] + X_eng['public speaking points']

X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42, stratify=y)
xgb2 = XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
xgb2.fit(X_train, y_train)
acc2 = accuracy_score(y_test, xgb2.predict(X_test))
print(f"  XGBoost Accuracy: {acc2*100:.2f}%")

print("\n--- Experiment 3: One-Hot Encoding ---")
df_onehot = df.drop(target_col, axis=1).copy()
df_onehot = pd.get_dummies(df_onehot, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(df_onehot, y, test_size=0.2, random_state=42, stratify=y)
xgb3 = XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
xgb3.fit(X_train, y_train)
acc3 = accuracy_score(y_test, xgb3.predict(X_test))
print(f"  XGBoost Accuracy: {acc3*100:.2f}%")

print("\n--- Experiment 4: Only Top 5 Features ---")
top5 = mi_df.head(5)['Feature'].tolist()
X_top5 = X_base[top5]
X_train, X_test, y_train, y_test = train_test_split(X_top5, y, test_size=0.2, random_state=42, stratify=y)
xgb4 = XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
xgb4.fit(X_train, y_train)
acc4 = accuracy_score(y_test, xgb4.predict(X_test))
print(f"  XGBoost Accuracy: {acc4*100:.2f}%")

print("\n--- Experiment 5: Reduced Classes (6 instead of 12) ---")
role_mapping = {
    'Applications Developer': 'Developer',
    'Software Developer': 'Developer',
    'Mobile Applications Developer': 'Developer',
    'Web Developer': 'Developer',
    'CRM Technical Developer': 'Developer',
    'Database Developer': 'Database/Backend',
    'Software Engineer': 'Engineer',
    'UX Designer': 'Designer/Frontend',
    'Network Security Engineer': 'Security',
    'Systems Security Administrator': 'Security',
    'Software Quality Assurance (QA) / Testing': 'QA/Support',
    'Technical Support': 'QA/Support',
}
y_grouped = df[target_col].map(role_mapping)
y_grouped_enc = LabelEncoder().fit_transform(y_grouped)
X_train, X_test, y_train, y_test = train_test_split(X_base, y_grouped_enc, test_size=0.2, random_state=42, stratify=y_grouped_enc)
xgb5 = XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
xgb5.fit(X_train, y_train)
acc5 = accuracy_score(y_test, xgb5.predict(X_test))
print(f"  XGBoost Accuracy: {acc5*100:.2f}%")
print(f"  (Random chance for 6 classes = {100/6:.1f}%)")

print("\n" + "=" * 60)
print("SUMMARY OF ALL EXPERIMENTS")
print("=" * 60)
print(f"  Random chance (12 classes):        {100/12:.2f}%")
print(f"  Exp 1 - Baseline Label Encoding:   {acc1*100:.2f}%")
print(f"  Exp 2 - Feature Engineering:       {acc2*100:.2f}%")
print(f"  Exp 3 - One-Hot Encoding:          {acc3*100:.2f}%")
print(f"  Exp 4 - Top 5 Features Only:       {acc4*100:.2f}%")
print(f"  Exp 5 - Reduced Classes (6):       {acc5*100:.2f}%")
print("=" * 60)
