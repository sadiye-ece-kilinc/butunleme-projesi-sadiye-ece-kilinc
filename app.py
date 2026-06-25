import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Sayfa başlığı ve tasarımı
st.set_page_config(page_title="CrediShield - Kredi Riski Analizi", layout="centered")

st.title(" CrediShield: Kredi Riski Karar Destek Sistemi")
st.write("Açıklanabilir Yapay Zekâ Destekli Risk Tahmin Modeli")
st.markdown("---")

# Model ve Öznitelik listesini yüklüyoruz
@st.cache_resource
def load_assets():
    model = joblib.load("kredi_xgb_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

try:
    xgb_model, model_features = load_assets()
    
    st.subheader(" Başvuru Sahibi Bilgileri")
    
    # Kullanıcıdan girdi toplama arayüzü
    person_age = st.number_input("Yaş", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Yıllık Gelir ($)", min_value=0, value=50000)
    person_emp_length = st.number_input("Çalışma Süresi (Yıl)", min_value=0, max_value=50, value=5)
    loan_amnt = st.number_input("Talep Edilen Kredi Miktarı ($)", min_value=0, value=10000)
    loan_int_rate = st.number_input("Kredi Faiz Oranı (%)", min_value=0.0, max_value=100.0, value=11.0)
    cb_person_cred_hist_length = st.number_input("Kredi Geçmişi Süresi (Yıl)", min_value=0, max_value=50, value=3)

    # Basitleştirilmiş One-Hot girdileri (Varsayılan olarak en sık kullanılanlar)
    loan_percent_income = loan_amnt / (person_income if person_income > 0 else 1)

    st.markdown("---")
    
    # Tahmin Butonu
    if st.button(" Kredi Riskini Hesapla"):
        # Modelin beklediği tüm kolonları sıfırla doldurarak bir veri çerçevesi oluşturuyoruz
        girdi_sozlugu = {col: [0] for col in model_features}
        
        # Kullanıcının girdilerini ilgili kolonlara yerleştiriyoruz
        girdi_sozlugu['person_age'] = [person_age]
        girdi_sozlugu['person_income'] = [person_income]
        girdi_sozlugu['person_emp_length'] = [person_emp_length]
        girdi_sozlugu['loan_amnt'] = [loan_amnt]
        girdi_sozlugu['loan_int_rate'] = [loan_int_rate]
        girdi_sozlugu['loan_percent_income'] = [loan_percent_income]
        girdi_sozlugu['cb_person_cred_hist_length'] = [cb_person_cred_hist_length]
        
        # Dataframe haline getirme
        girdi_df = pd.DataFrame(girdi_sozlugu)
        
        # Model tahmini
        risk_ihtimali = xgb_model.predict_proba(girdi_df)[0][1]
        tahmin_sinifi = xgb_model.predict(girdi_df)[0]
        
        # Sonuç ekranı
        st.subheader(" Analiz Sonucu")
        if tahmin_sinifi == 1:
            st.error(st.error(f" **YÜKSEK RİSK:** Kredi başvurusu riskli görünüyor. Temerrüt İhtimali: %{risk_ihtimali*100:.2f}"))
        else:
            st.success(f" **DÜŞÜK RİSK:** Kredi başvurusu onaylanabilir. Temerrüt İhtimali: %{risk_ihtimali*100:.2f}")
            
        st.markdown("---")
        st.subheader(" Karar Gerekçesi (Açıklanabilirlik)")
        st.write("Modelin bu kararı verirken veri setindeki genel özniteliklerden nasıl etkilendiğini gösteren SHAP grafiği aşağıdadır:")
        # Daha önce kaydettiğimiz SHAP grafiğini ekranda gösteriyoruz
        st.image("shap_summary_plot.png", caption="Genel Karar Mekanizması (SHAP Özet Grafiği)")

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")