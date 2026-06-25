import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib

print(" Adım 1: Eğitilmiş model ve veriler yükleniyor...")
# Kaydettiğimiz modeli ve özellikleri geri yüklüyoruz
xgb_model = joblib.load("kredi_xgb_model.pkl")
features = joblib.load("model_features.pkl")

# Test etmek için temiz veri setini yüklüyoruz
df = pd.read_csv("credit_risk_clean.csv")
X = df[features]

print(" Adım 2: SHAP Explainer (Açıklayıcı) oluşturuluyor...")
# XGBoost modeli için SHAP ağaç açıklayıcısını kuruyoruz
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X)

print(" Adım 3: Model kararlarını etkileyen genel faktörler grafikleştiriliyor...")
# Grafiklerin düzgün çizilmesi için matplotlib ayarı yapıyoruz
plt.figure(figsize=(10, 6))

# Genel olarak kredi riskini en çok etkileyen özellikleri gösteren özet grafik (Summary Plot)
shap.summary_plot(shap_values, X, show=False)

# Grafik dosyasını bilgisayara kaydediyoruz (Arayüzde ve raporda kullanmak için)
graph_name = "shap_summary_plot.png"
plt.tight_layout()
plt.savefig(graph_name, dpi=300)
plt.close()

print(f" Başarılı! SHAP özet grafiği '{graph_name}' adasıyla klasörünüze kaydedildi.")