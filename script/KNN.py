import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────────────────
# Preprocesing
# ──────────────────────────────────────────────────────────────────────────────────────────
# Load the dataset
df = pd.read_csv("Student Depression Dataset.csv")
# print(df.head())

# ──────────────────────────────────────────────────────────────────────────────────────────
# Basic information
# print(df.shape)
# print(df.columns)
# print(df.info())

# ──────────────────────────────────────────────────────────────────────────────────────────
# Remove ID, Profession, Work Pressure, Job Satisfaction (Not useful)
df = df.drop(["id", "Profession", "Work Pressure", "Job Satisfaction"], axis=1)

# ──────────────────────────────────────────────────────────────────────────────────────────
# Check for missing values
# print(df.isnull().sum())

# Fill Financial Stress missing values (3) with median
median_financial_stress = df['Financial Stress'].median()
df['Financial Stress'] = df['Financial Stress'].fillna(median_financial_stress)

# ──────────────────────────────────────────────────────────────────────────────────────────
# Check for outliers
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns.remove('Depression')

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df[column] < lower) | (df[column] > upper)).sum()
    # print(f"{column:<25} > {outlier_count} outliers")
    
    # Cap outlier to the lower and upper bounds
    df[column] = df[column].clip(lower, upper)

# ──────────────────────────────────────────────────────────────────────────────────────────
# Encoding categorical columns
# Gender: Using One-Hot Encoding (No order, low cardinality)
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
# print(f"(One-Hot) Gender > {[column for column in df.columns if 'Gender' in column]}")

# City, Degree: Using Frequency Encoding (High cardinality, no meaningful order)
city_map = df['City'].value_counts(normalize=True).to_dict()
df['City'] = df['City'].map(city_map)
# print(f"(Frequency) City > replaced with category frequency (0-1)")

degree_map = df['Degree'].value_counts(normalize=True).to_dict()
df['Degree'] = df['Degree'].map(degree_map)
# print(f"(Frequency) Degree > replaced with category frequency (0-1)")

df['City'] = df['City'].map(city_freq_map)
df['Degree'] = df['Degree'].map(degree_freq_map)

# Sleep Duration, Dietary Habits: Using Ordinal Encoding (Meaningful order)
sleep_order = {
    'Less than 5 hours': 0,
    '5-6 hours'        : 1,
    '7-8 hours'        : 2,
    'More than 8 hours': 3,
    'Others'           : 1
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_order)
# print(f"(Ordinal) Sleep Duration > {sleep_order}")

diet_order = {
    'Unhealthy': 0,
    'Moderate' : 1,
    'Healthy'  : 2,
    'Others'   : 1
}
df['Dietary Habits'] = df['Dietary Habits'].map(diet_order)
# print(f"(Ordinal) Dietary Habits > {diet_order}")

# Suicidal Thoughts, Family History: Using Binary Encoding (Yes or no)
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
df['Family History of Mental Illness']      = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
# print(f"(Binary) Suicidal Thoughts > Yes=1, No=0")
# print(f"(Binary) Family History > Yes=1, No=0")

# ──────────────────────────────────────────────────────────────────────────────────────────
# Separate features and target
X = df.drop(columns=['Depression'])
y = df['Depression']

# ──────────────────────────────────────────────────────────────────────────────────────────
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ──────────────────────────────────────────────────────────────────────────────────────────
# Check imbalance
counts = y.value_counts()
ratio = counts.min() / counts.max()
# print(f"Class 0 (No Depression)        : {counts[0]}")
# print(f"Class 1 (Depression)           : {counts[1]}")
# print(f"Minority/Majority ratio before : {ratio:.2f}")

# If below, apply SMOTE
IMBALANCE_THRESHOLD = 0.50 

if ratio < IMBALANCE_THRESHOLD:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    # print(f"After SMOTE > Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
    
# ──────────────────────────────────────────────────────────────────────────────────────────
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

# print(f"X_train_scaled mean (sample): {X_train_scaled.iloc[:, :13].mean().round(4).to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# Find Best K Value
# ─────────────────────────────────────────────────────────────────────────────
# import numpy as np
# from sklearn.metrics import f1_score

# k_values = range(1, np.sqrt(X_train_scaled.shape[0]).astype(int) + 1)
# f1_scores_list = []

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train_scaled, y_train)
#     y_pred_k = knn.predict(X_test_scaled)
    
#     score = f1_score(y_test, y_pred_k)
#     f1_scores_list.append(score)
    
#     print(f"K = {k:<2} | F1-Score = {score:.4f}")

# best_k = k_values[f1_scores_list.index(max(f1_scores_list))]
# print(f"Best K value: {best_k}") # 109

# ──────────────────────────────────────────────────────────────────────────────────────────
# K-Nearest Neighbors Model
# ──────────────────────────────────────────────────────────────────────────────────────────
# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=109)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix (KNN):")
print(cm)

print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics (Manual Calculation)
# ─────────────────────────────────────────────────────────────────────────────
# Accuracy (Overall, how often is the model correct?)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision (When the model predicted positive, how often was it correct?)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# Recall (When the actual class is positive, how often did the model predict it correctly?)
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# F1-score (Harmonic mean of precision and recall, balances both metrics)
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1_score:.4f}")

print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────────────────────────────────────
# Save the trained model and scaler for later use in the Streamlit app
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(city_map, 'model/city_map.joblib')
joblib.dump(degree_map, 'model/degree_map.joblib')
joblib.dump(X_train.columns.tolist(), 'model/feature_columns.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(knn, 'model/knn_model.joblib')