# 🏗️ Steel Plate Defect Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-black.svg?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## 📝 Proje Açıklaması
Bu proje, endüstriyel çelik levha üretim hatlarındaki yüzey kusurlarını makine öğrenmesi kullanarak yüksek hassasiyetle sınıflandırmak amacıyla geliştirilmiştir. Karmaşık doku ve geometrik özellikler analiz edilerek kalite kontrol süreçlerinin otomatize edilmesi hedeflenmiştir.

## 🎯 Ana Hedef
*   Endüstriyel kalite kontrol süreçlerini hatasız ve hızlı bir şekilde otomatikleştirmek.
*   Üretim hattındaki hataları gerçek zamanlı olarak sınıflandırmak.

## 🚀 Model Performansı
*   **Kaggle AUC Score:** `0.8899`
*   **Optimizasyon:** Optuna ile hiperparametre ayarları yapılmış ve en iyi "beton karma" değerleri (Learning Rate: 0.01, Max Depth: 6) belirlenmiştir.

## 🛠️ Özellik Mühendisliği (Feature Engineering)
Modelin ayırt ediciliğini artırmak için ham verilere eklenen teknik metrikler:
*   **Aspect Ratio:** Kusurların en-boy oranı analizi.
*   **Luminosity Range:** Yüzeydeki parlaklık farkları üzerinden doku analizi.
*   **Shape Features:** Geometrik form kısıtları.
*   **Global Index Metrics:** Kusurun levha üzerindeki konumsal ağırlığı.

## 💻 Kurulum ve Çalıştırma
Projeyi yerel ortamınızda çalıştırmak için:
```bash
# Gereksinimleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run app.py
```
## 📂 Gerekli Dosyalar

Projenin çalışması için dizinde bulunması gereken kritik bileşenler:

| Dosya Adı | Açıklama |
| :--- | :--- |
| **optimized_xgboost_model.pkl** | Eğitilmiş ve mühürlenmiş nihai XGBoost modeli. |
| **scaler.pkl** | Veri standardizasyonu için kullanılan ölçekleyici. |
| **train.csv / test.csv** | Model eğitimi ve test aşamaları için kullanılan veri setleri. |

## 📊 Streamlit Arayüz Özellikleri

Kullanıcı dostu arayüz üzerinden sunulan teknik imkanlar:

*   **Veri Girişi:** Tekli manuel veri girişi veya toplu CSV dosyası yükleme desteği.
*   **Tahmin Mekanizması:** 7 farklı hata türü için hesaplanan detaylı olasılık skorları.
*   **Görsel Analiz:** Modelin karar mekanizmasını gösteren interaktif **Feature Importance** grafikleri.

## 🛠️ Gelecek Geliştirmeler

Projenin "faz 2" aşaması için planlanan teknik iyileştirmeler:

*   **SHAP Analizi:** Model kararlarının şeffaflaştırılması ve her tahminin nedeninin açıklanması.
*   **Ensemble Modeller:** Farklı algoritmaların birleştirilerek skorun **0.90+** bandına taşınması.
*   **Endüstriyel Entegrasyon:** Docker konteynerizasyon ve API desteği ile gerçek üretim hatlarına dağıtım.

---

## 👷 Sorumlu Mühendis

**Serdar Önal**  
*20 Yıllık Deneyimli İnşaat Mühendisi | AI & Data Science Developer*

> **Not:** Bu çalışma, 30 Mayıs 2026 hedefli yapay zeka müfredatı kapsamında, mühendislik disiplini ve titizliğiyle geliştirilmiştir.