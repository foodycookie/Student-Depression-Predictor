import pandas as pd

df = pd.read_csv("Student Depression Dataset.csv")

print("age")
print(df['Age'].min())
print(df['Age'].max())

print("academic pressure")
print(df['Academic Pressure'].min())
print(df['Academic Pressure'].max())
print(df['Academic Pressure'].unique())

print("CGPA")
print(df['CGPA'].min())
print(df['CGPA'].max())
print(df['CGPA'].unique())

print("Study Satisfaction")
print(df['Study Satisfaction'].min())
print(df['Study Satisfaction'].max())
print(df['Study Satisfaction'].unique())

print("Work/Study Hours")
print(df['Work/Study Hours'].min())
print(df['Work/Study Hours'].max())
print(df['Work/Study Hours'].unique())

print("Financial Stress")
print(df['Financial Stress'].min())
print(df['Financial Stress'].max())
print(df['Financial Stress'].unique())