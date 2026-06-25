import pandas as pd
import numpy as np

print("Adım 1: Veri seti yükleniyor...")
# Klasördeki veri setini yüklenir
df = pd.read_csv("credit_risk_dataset.csv")

print("\n İlk 5 Satır:")
print(df.head())

print("\n Eksik Veri Durumu:")
print(df.isnull().sum())

print("\n Adım 2: Eksik veriler temizleniyor/dolduruluyor...")
# Kişinin çalışma süresi eksikse, orta (medyan) değer ile doldurulur
emp_length_median = df['person_emp_length'].median()
df['person_emp_length'] = df['person_emp_length'].fillna(emp_length_median)

# Faiz oranı eksikse, yine orta faiz oranı ile doldurulur
loan_int_median = df['loan_int_rate'].median()
df['loan_int_rate'] = df['loan_int_rate'].fillna(loan_int_median)

print("\n Adım 3: Kategorik (Metin) veriler sayısal değerlere dönüştürülüyor...")
# Modellerin metinleri anlayabilmesi için sayısal verilere çevrilir
kategorik_kolonlar = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df_clean = pd.get_dummies(df, columns=kategorik_kolonlar, drop_first=True)

# Temizlenmiş veri kaydedilir
df_clean.to_csv("credit_risk_clean.csv", index=False)
print("\n Temizlenmiş veri 'credit_risk_clean.csv' olarak kaydedildi!")