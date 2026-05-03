import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Steel Plate Defect Prediction", layout="wide")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model():
    # Az önce mühürlediğimiz modeli çağırıyoruz
    return joblib.load('optimized_xgboost_model.pkl')

model = load_model()

# --- SABİTLER ---
CLASS_NAMES = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
FEATURE_NAMES = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
]

# --- ÖZELLİK MÜHENDİSLİĞİ (ENGINEER FEATURES) ---
# Modelin beklediği tüm 'malzeme listesini' burada tanımlıyoruz
def engineer_features(df):
    df_new = df.copy()
    
    # 1. Defect Width & Height
    df_new['Defect_Width'] = df_new['X_Maximum'] - df_new['X_Minimum']
    df_new['Defect_Height'] = df_new['Y_Maximum'] - df_new['Y_Minimum']
    
    # 2. Geometrik Oranlar
    df_new['Aspect_Ratio'] = df_new['Defect_Width'] / (df_new['Defect_Height'] + 1e-5)
    df_new['Area_Perimeter_Ratio'] = df_new['Pixels_Areas'] / (df_new['X_Perimeter'] + df_new['Y_Perimeter'] + 1e-5)
    
    # 3. Işık ve Doku Özellikleri (Raporundaki doku karmaşası analizi için kritik)
    df_new['Luminosity_Range'] = df_new['Maximum_of_Luminosity'] - df_new['Minimum_of_Luminosity']
    df_new['Luminosity_Density'] = df_new['Sum_of_Luminosity'] / (df_new['Pixels_Areas'] + 1e-5)
    
    return df_new

# --- UI TASARIMI ---
st.title("🏗️ Steel Plate Defect Prediction")
st.markdown("### Serdar Önal | 20 Yıllık Mühendislik Deneyimi ile AI Çözümleri")
st.markdown("---")

# Yan Panel
mode = st.sidebar.radio("📋 Tahmin Modu Seçin", ["Tekli Tahmin", "Toplu CSV Tahmini"])

if mode == "Tekli Tahmin":
    st.subheader("📝 Manuel Veri Girişi")
    user_data = {}
    
    # Veri giriş kutularını 3 sütunlu yapalım
    cols = st.columns(3)
    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            user_data[feature] = st.number_input(feature, value=0.0)

    if st.button("🚀 Tahmin Yap"):
        # Veriyi DataFrame yap ve mühendislik sütunlarını ekle
        input_df = pd.DataFrame([user_data])
        input_df_final = engineer_features(input_df)
        
        # TAHMİN
        raw_pred = model.predict(input_df_final)
        
        # GÜVENLİ DÖNÜŞÜM: 
        # Önce düz bir listeye çeviriyoruz (.flatten()), 
        # sonra ilk elemanı ([0]) alıp tam sayıya (int) dönüştürüyoruz.
        prediction = int(np.array(raw_pred).flatten()[0])
        
        probabilities = model.predict_proba(input_df_final)[0]

        st.success(f"🎯 Tahmin Edilen Kusur: **{CLASS_NAMES[prediction]}**")

        # Olasılık Grafiği
        prob_df = pd.DataFrame({"Kusur Türü": CLASS_NAMES, "Olasılık": probabilities})
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=prob_df, x="Olasılık", y="Kusur Türü", palette="viridis", ax=ax)
        plt.title("Sınıf Bazlı Tahmin Olasılıkları")
        st.pyplot(fig)

elif mode == "Toplu CSV Tahmini":
    st.subheader("📂 CSV Dosyası Yükleyin")
    uploaded_file = st.file_uploader("Kusur analizi yapılacak CSV'yi seçin", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 Yüklenen Veri Önizlemesi:", df.head())

        # İşlem adımları
        engineered_df = engineer_features(df)
        predictions = model.predict(engineered_df)
        probabilities = model.predict_proba(engineered_df)

        # Sonuçları işle (int dönüşümü ile)
        results = df.copy()
        results["Predicted_Class"] = [CLASS_NAMES[int(p)] for p in predictions]
        
        for i, class_name in enumerate(CLASS_NAMES):
            results[f"Prob_{class_name}"] = probabilities[:, i]

        st.success("✅ Tüm levha tahminleri başarıyla tamamlandı!")
        st.dataframe(results.head())

        # İndirme
        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Analiz Sonuçlarını İndir", csv_data, "analiz_sonuclari.csv", "text/csv")

# --- MODEL ÖNEM ANALİZİ ---
st.markdown("---")
st.subheader("📊 Karar Mekanizması (Feature Importance)")
if hasattr(model, "feature_importances_"):
    # Mühendisliği yapılmış kolonları baz alalım
    temp_df = engineer_features(pd.DataFrame(columns=FEATURE_NAMES))
    importance_df = pd.DataFrame({
        "Feature": temp_df.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="magma", ax=ax)
    st.pyplot(fig)

st.caption("© 2026 Serdar Önal - 30 Mayıs Final Projesi Teslimat Dosyası")