# ----------------------------------------------------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------------------------------------------------

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

os.makedirs("model", exist_ok=True)
os.makedirs("photo", exist_ok=True)

warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------------------------------------------------------------
# Data Preparation
# ----------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('Student Depression Dataset.csv')

print("Dataset Overview")
print(df)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ----------------------------------------------------------------------------------------------------------------------------------

df = df.drop(["id", "City"], axis=1)
print("\nDataset Overview after id and City are excluded:")
print(df)

print("\nMissing values in each column:")
print(df.isnull().sum())

median_financial_stress = df['Financial Stress'].median()
df['Financial Stress'] = df['Financial Stress'].fillna(median_financial_stress)
print("\nReplaced missing values in 'Financial Stress' with median:", median_financial_stress)

# ----------------------------------------------------------------------------------------------------------------------------------

print("\nSummary statistics for numerical columns:")
print()
print(df['Age'].describe())
print()
print(df['CGPA'].describe())
print()
print(df['Work/Study Hours'].describe())

# Outlier detection using IQR method
print("\nPotential outliers (IQR method):")

numeric_cols = ['Age', 'CGPA', 'Work/Study Hours']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"{col}:")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Lower Bound={lower_bound:.2f}, Upper Bound={upper_bound:.2f}")
    print(f"  Potential outliers: {len(outliers)}")
    print()

print("Unique values in categorical columns:")
print()
print(df['Gender'].unique())
print()
print(df['Profession'].unique())
print()
print(df['Academic Pressure'].unique())
print()
print(df['Work Pressure'].unique())
print()
print(df['Study Satisfaction'].unique())
print()
print(df['Job Satisfaction'].unique())
print()
print(df['Sleep Duration'].unique())
print()
print(df['Dietary Habits'].unique())
print()
print(df['Degree'].unique())
print()
print(df['Have you ever had suicidal thoughts ?'].unique())
print()
print(df['Financial Stress'].unique())
print()
print(df['Family History of Mental Illness'].unique())
print()
print(df['Depression'].unique())

# ----------------------------------------------------------------------------------------------------------------------------------

# Group Degrees into Education Levels
def map_degree(deg):
    if deg == 'Class 12':
        return 'School'
    
    elif deg in ['B.Pharm','BSc','BA','BCA','B.Ed','LLB','BE','BHM','B.Com','B.Arch','B.Tech','BBA']:
        return 'Bachelor'
    
    elif deg in ['M.Tech','M.Ed','MSc','M.Pharm','MCA','MA','MBA','M.Com','ME','MHM','LLM']:
        return 'Master'
    
    elif deg in ['PhD','MD','MBBS']:
        return 'Doctorate'
    
    else:
        return 'Others'

df['Degree_Level'] = df['Degree'].apply(map_degree)
df = df.drop(["Degree"], axis=1)

# ----------------------------------------------------------------------------------------------------------------------------------

# Correlation Analysis

numeric_cols = ["Age", "CGPA", "Work/Study Hours"]
categorical_cols = ["Gender", "Profession", "Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction", "Sleep Duration", "Dietary Habits", "Degree_Level", "Have you ever had suicidal thoughts ?", "Financial Stress", "Family History of Mental Illness", "Depression"]

print("\nNumeric Correlation (Pearson)")
print(df[numeric_cols].corr().round(2).to_string())

def cramers_v(col1, col2):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

print("\nCategorical Correlation (Cramér's V)")
cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
print(cramers_matrix.round(2).to_string())

def eta_squared(num_col, cat_col):
    groups = [group.values for _, group in num_col.groupby(cat_col)]
    grand_mean = num_col.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return ss_between / ss_total if ss_total != 0 else 0

print("\nNumeric-Categorical Correlation (Eta Squared)")
eta_matrix = pd.DataFrame(index=numeric_cols, columns=categorical_cols, dtype=float)
for num in numeric_cols:
    for cat in categorical_cols:
        eta_matrix.loc[num, cat] = eta_squared(df[num], df[cat])
print(eta_matrix.round(2).to_string())

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
fig.suptitle('Correlation Analysis', fontsize=18, fontweight='bold')

# Pearson heatmap
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=axes[0], annot_kws={"size": 10})
axes[0].set_title("Numeric–Numeric\n(Pearson)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Cramér's V heatmap
sns.heatmap(cramers_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[1], annot_kws={"size": 7})
axes[1].set_title("Categorical–Categorical\n(Cramér's V)", fontsize=12)
axes[1].tick_params(axis='x', rotation=90)
axes[1].tick_params(axis='y', rotation=0)

# Eta squared heatmap
sns.heatmap(eta_matrix.astype(float), annot=True, fmt='.2f', cmap='BuGn',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[2], annot_kws={"size": 7})
axes[2].set_title("Numeric–Categorical\n(Eta Squared)", fontsize=12)
axes[2].tick_params(axis='x', rotation=90)
axes[2].tick_params(axis='y', rotation=0)

plt.tight_layout(pad=3.0)
_fname = "photo/correlation_analysis.png"
plt.savefig(_fname, bbox_inches='tight', dpi=300)
plt.close()
print(f'\nFile saved "{_fname}"')

# ----------------------------------------------------------------------------------------------------------------------------------
# Data Transformation and Encoding
# ----------------------------------------------------------------------------------------------------------------------------------

print(f"\nShape before encoding: {df.shape}")

# One-Hot Encoding
df = pd.get_dummies(df, columns=["Profession"], drop_first=True)

# Ordinal Encoding
df["Sleep Duration"]  = df["Sleep Duration"].map({"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3, "Others": 1})
df["Dietary Habits"] = df["Dietary Habits"].map({"Unhealthy": 0, "Moderate": 1, "Healthy": 2, "Others": 1})
df["Degree_Level"] = df["Degree_Level"].map({"School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3, "Others": 4})

# Binary Encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

print(f"\nShape after encoding: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

print(f"\nSleep Duration unique values: {sorted(df['Sleep Duration'].unique())}")
print(f"\nDietary Habits unique values: {sorted(df['Dietary Habits'].unique())}")
print(f"\nDegree_Level unique values: {sorted(df['Degree_Level'].unique())}")
print(f"\nGender unique values: {sorted(df['Gender'].unique())}")
print(f"\nSuicidal Thoughts unique values: {sorted(df['Have you ever had suicidal thoughts ?'].unique())}")
print(f"\nFamily History unique values: {sorted(df['Family History of Mental Illness'].unique())}")

# ----------------------------------------------------------------------------------------------------------------------------------

# Train-Test Split
X = df.drop(columns=['Depression'])
y = df['Depression']

print("\nBefore Split:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y dtype: {y.dtype}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nAfter Split:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Save feature columns
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, "model/feature_columns.pkl")

# ----------------------------------------------------------------------------------------------------------------------------------

# Class Imbalance Check
print("\nClass Imbalance Check")

counts = y.value_counts()
ratio = counts.min() / counts.max()

print(f"Class 0 (No Depression)        : {counts[0]}")
print(f"Class 1 (Depression)           : {counts[1]}")
print(f"Minority/Majority ratio before : {ratio:.2f}")

# If below, apply SMOTE
IMBALANCE_THRESHOLD = 0.50 

if ratio < IMBALANCE_THRESHOLD:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE > Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
    
else:
    print("No SMOTE needed, classes are sufficiently balanced")

# ----------------------------------------------------------------------------------------------------------------------------------
# Modelling and Parameter Tuning (Already done, so commented out for faster execution)
# ----------------------------------------------------------------------------------------------------------------------------------

# print("\nModel 1: Decision Tree")

# dt_params = {
#     'max_depth':        [3, 5, 10, 15, 20, None],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf':  [1, 2, 5, 10],
#     'criterion':        ['gini', 'entropy']
# }

# dt_base = DecisionTreeClassifier(random_state=42)

# dt_search = RandomizedSearchCV(
#     dt_base, dt_params,
#     n_iter=20, cv=5,
#     scoring='accuracy',
#     random_state=42, n_jobs=-1, verbose=1
# )

# dt_search.fit(X_train, y_train)

# dt_model = dt_search.best_estimator_
# print(f"Best params: {dt_search.best_params_}")
# print(f"Best CV accuracy: {dt_search.best_score_:.4f}")

# # ----------------------------------------------------------------------------------------------------------------------------------

# print("\nModel 2: Random Forest")

# rf_params = {
#     # number of trees
#     'n_estimators': [100, 200, 300],
#     # max tree depth
#     'max_depth': [None, 10, 20, 30],
#     # min samples to split a node
#     'min_samples_split': [2, 5, 10],
#     # min samples at leaf node
#     'min_samples_leaf': [1, 2, 4],
#     # features considered per split
#     'max_features': ['sqrt', 'log2']
# }

# rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# rf_search = RandomizedSearchCV(
#     rf_base, rf_params,
#     n_iter=20, cv=5,
#     scoring='accuracy',
#     random_state=42, n_jobs=-1, verbose=1
# )

# rf_search.fit(X_train, y_train)   

# rf_model = rf_search.best_estimator_
# print(f"Best params: {rf_search.best_params_}")
# print(f"Best CV accuracy: {rf_search.best_score_:.4f}")

# # ----------------------------------------------------------------------------------------------------------------------------------
# print("\nModel 3: XGBoost")

# xgb_params = {
#     # number of boosting rounds
#     'n_estimators': [100, 200, 300],
#     # tree depth
#     'max_depth': [3, 5, 7, 9],
#     # step size shrinkage
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     # row sampling per tree
#     'subsample': [0.7, 0.8, 1.0],
#     # feature sampling per tree
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }

# xgb_base = XGBClassifier(
#     objective='multi:softmax',
#     num_class=3,
#     eval_metric='mlogloss',
#     random_state=42,
#     n_jobs=-1,
#     verbosity=0
# )

# xgb_search = RandomizedSearchCV(
#     xgb_base, xgb_params,
#     n_iter=20, cv=5,
#     scoring='accuracy',
#     random_state=42, n_jobs=-1, verbose=1
# )

# xgb_search.fit(X_train, y_train)   

# xgb_model = xgb_search.best_estimator_
# print(f"Best params: {xgb_search.best_params_}")
# print(f"Best CV accuracy: {xgb_search.best_score_:.4f}")

# ----------------------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------------------

# Models after parameter tuning 
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion='entropy', max_depth=5, min_samples_leaf=1, min_samples_split=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_split=2, min_samples_leaf=4,
        max_features='sqrt', max_depth=30, random_state=42
    ),
    "XGBoost": XGBClassifier(
        subsample=0.7, n_estimators=200, max_depth=3,
        learning_rate=0.05, colsample_bytree=0.7,
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )
}

# ----------------------------------------------------------------------------------------------------------------------------------

label_names = ['No Depression', 'Depression']
results = {}

# Fit and Evaluate
for name, model in models.items():
    print(f"\nModel: {name}")
    
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")
    
    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall    = recall_score(y_test, y_pred, average='binary')
    f1        = f1_score(y_test, y_pred, average='binary')

    report = classification_report(y_test, y_pred, target_names=label_names)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "accuracy" : acc,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1
    }

# ----------------------------------------------------------------------------------------------------------------------------------

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices', fontsize=15, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAcc: {res['accuracy']:.4f}", fontsize=11, fontweight='bold')

plt.tight_layout()
_fname = "photo/confusion_matrix.png"
plt.savefig(_fname, bbox_inches='tight')
plt.close()
print(f'\nFile saved "{_fname}"')

# ----------------------------------------------------------------------------------------------------------------------------------

# Bar Chart
metrics = ['accuracy', 'precision', 'recall', 'f1']
titles  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
names   = list(results.keys())

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')

for ax, metric, title in zip(axes, metrics, titles):
    values = [results[n][metric] for n in names]
    bars   = ax.bar(names, values, edgecolor='white', width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylim(0.5, 1.05)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(title)
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
_fname = "photo/model_performance_comparison.png"
plt.savefig(_fname, bbox_inches='tight')
plt.close()
print(f'\nFile saved "{_fname}"')

# ----------------------------------------------------------------------------------------------------------------------------------

# Summary
print("\nFinal Summary\n")

# CV Accuracy vs Test Accuracy
print(f"{'Model':<25} {'CV Accuracy':>12} {'Test Accuracy':>14}")
print("=" * 55)
cv_scores = {"Decision Tree": 0.8280, "Random Forest": 0.8457, "XGBoost": 0.8476}
for name in results:
    print(f"{name:<25} {cv_scores[name]:>12.4f} {results[name]['accuracy']:>14.4f}")

print()

# Summarize the big 4
print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
print("=" * 65)
for name in results:
    r = results[name]
    print(f"{name:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")
print("=" * 65)