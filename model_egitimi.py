import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print(" Adım 1: Temizlenmiş veri seti yükleniyor...")
df = pd.read_csv("credit_risk_clean.csv")

# Yapay zekaya neyi tahmin edeceğini söylüyoruz: loan_status (Kredi Durumu: 1=Riskli, 0=Güvenli)
X = df.drop(columns=['loan_status'])
y = df['loan_status']

print(" Adım 2: Veri %80 Eğitim, %20 Test olarak ikiye ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------------------------------
# 1. MODEL: Random Forest (Rastgele Orman)
# ----------------------------------------------------
print("\n Adım 3: 1. Model (Random Forest) eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Modelin başarısını ölçüyoruz
rf_train_preds = rf_model.predict(X_train)
rf_test_preds = rf_model.predict(X_test)

print(f" Random Forest - Eğitim Doğruluğu: %{accuracy_score(y_train, rf_train_preds)*100:.2f}")
print(f" Random Forest - Test Doğruluğu: %{accuracy_score(y_test, rf_test_preds)*100:.2f}")

# ----------------------------------------------------
# 2. MODEL: XGBoost (Gelişmiş Gradyan Artırma)
# ----------------------------------------------------
print("\n Adım 4: 2. Model (XGBoost) eğitiliyor...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Modelin başarısını ölçüyoruz
xgb_train_preds = xgb_model.predict(X_train)
xgb_test_preds = xgb_model.predict(X_test)

print(f" XGBoost - Eğitim Doğruluğu: %{accuracy_score(y_train, xgb_train_preds)*100:.2f}")
print(f" XGBoost - Test Doğruluğu: %{accuracy_score(y_test, xgb_test_preds)*100:.2f}")

# ----------------------------------------------------
# EN BAŞARILI MODELİ KAYDETME
# ----------------------------------------------------
print("\n Adım 5: Modeller gelecekteki arayüz için bilgisayara kaydediliyor...")
joblib.dump(xgb_model, "kredi_xgb_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_features.pkl")
print(" 'kredi_xgb_model.pkl' başarıyla kaydedildi!")