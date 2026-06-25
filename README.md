#  CrediShield: Açıklanabilir Yapay Zekâ Destekli Kredi Riski Karar Destek Sistemi

Bu proje, Akademik Yapay Zekaya Giriş dersi Bireysel Ürün Geliştirme Projesi (Bütünleme Sınavı) kapsamında **"SEÇENEK 3 - Açıklanabilir Makine Öğrenmesi Karar Destek Ürünü"** olarak geliştirilmiştir.

---

##  Problem Tanımı
Finansal kuruluşlar, kredi başvurusunda bulunan bireylerin temerrüt (borcu geri ödeyememe) risklerini tahmin etmekte geleneksel ve statik yöntemler kullanmaktadır. Bu durum hem hatalı risk öngörülerine yol açmakta hem de yapay zekâ kararlarının arkasındaki gerekçelerin şeffaf bir şekilde finansal analistlere sunulamamasına (kara kutu problemi) neden olmaktadır. 

##  Hedef Kullanıcı
- Bankaların ve finans kuruluşlarının kredi onay departmanı uzmanları
- Risk yönetimi analistleri
- Finansal danışmanlar

##  Çözümün Kısa Açıklaması
CrediShield; Kaggle üzerinden alınan gerçekçi kredi riski veri setini işleyerek makine öğrenmesi modelleriyle (Random Forest ve XGBoost) eğitilmiş bir karar destek sistemidir. Geliştirilen ürün, son kullanıcıya sadece anlık bir risk skoru ve tahmini vermekle kalmaz; **SHAP (Açıklanabilir Yapay Zekâ - XAI)** entegrasyonu sayesinde kararın arkasındaki gerekçeleri görsel olarak açıklar.

---

##  Kullanılan Teknolojiler
- **Programlama Dili:** Python
- **Veri Analizi & Ön İşleme:** Pandas, NumPy
- **Makine Öğrenmesi Modelleri:** Scikit-Learn (Random Forest), XGBoost
- **Açıklanabilir Yapay Zekâ (XAI):** SHAP
- **Kullanıcı Arayüzü:** Streamlit
- **Proje Yönetimi:** GitHub (Issues & Releases)

---

##  Sistem Mimarisi ve İş Akışı
1. **Veri Toplama:** Kaggle'dan alınan ham kredi riski verilerinin yüklenmesi.
2. **Veri Temizleme & Hazırlama:** Eksik verilerin medyan ile doldurulması ve kategorik alanların One-Hot Encoding ile sayısal verilere dönüştürülmesi.
3. **Model Eğitimi:** Random Forest ve XGBoost modellerinin %80 eğitim ve %20 test verisi ayrımıyla eğitilmesi, overfitting kontrolü.
4. **Açıklanabilirlik (XAI):** Eğitilen en başarılı modelin kararlarının SHAP ağaç açıklayıcısı ile analiz edilerek küresel öznitelik grafiklerinin üretilmesi.
5. **Kullanıcı Arayüzü:** Finans analistinin parametreleri girerek tahmini ve SHAP gerekçesini gördüğü web arayüzünün Streamlit ile ayağa kaldırılması.

---

##  Kurulum ve Çalıştırma Adımları

### 1. Depoyu Klonlayın veya İndirin
```bash
git clone [https://github.com/sadiye-ece-kilinc/butunleme-projesi-sadiye-ece-kilinc.git](https://github.com/sadiye-ece-kilinc/butunleme-projesi-sadiye-ece-kilinc.git)
cd butunleme-projesi-sadiye-ece-kilinc
