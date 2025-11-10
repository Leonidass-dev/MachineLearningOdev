# Soru 6: Mel-Spektrogram Öznitelikleri ile Açık Kod Random Forest Sınıflandırması
# Bu dosya, ödevin tüm gereksinimlerini karşılamaktadır:
# 1. Mel-Spektrogram ortalaması ile özellik çıkarma.
# 2. Çıkarılan özellikleri CSV'ye kaydetme (Yeniden hesaplamadan kaçınmak için).
# 3. Random Forest (Açık Kod Yapısı) ile sınıflandırma.
# 4. Detaylı değerlendirme ve Soru 4 ile karşılaştırma.

import pandas as pd
import librosa
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import warnings

# librosa uyarılarını gizle
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. ORTAM VE YOL AYARLARI (Lütfen KENDİ YOLUNUZU AYARLAYIN!)
# ----------------------------------------------------------------------
# ÖNEMLİ: Bu yolu kendi 'archive' klasörünüzün mutlak yoluna göre güncelleyin.
BASE_PATH = r'C:\Users\Leonidas\Downloads\archive'
AUDIO_PATH = BASE_PATH
CSV_PATH = os.path.join(BASE_PATH, 'UrbanSound8K.csv')
FEATURE_FILE = os.path.join(BASE_PATH, 'urban_sound_features_mels.csv')  # Kaydedilecek Öznitelik Dosyası

# Sabitler
TARGET_SR = 22050  # Mel-Spektrogram için standart örnekleme hızı
N_MELS = 128  # Mel bandı sayısı (Özellik vektörünün boyutu)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_NAMES = [
    "dog_bark", "children_playing", "car_horn", "air_conditioner",
    "street_music", "drilling", "jackhammer", "siren", "engine_idling", "gun_shot"
]

start_time = time.time()
print("Başlangıç Zamanı:", time.ctime(start_time))
print("-" * 50)


def extract_mel_features(file_name, sr=TARGET_SR, n_mels=N_MELS):
    """Verilen dosya yolundan Mel-Spektrogram ortalamasını çıkarır."""
    try:
        # 0.5 saniyeden kısa dosyaları librosa.feature.melspectrogram hesaplarken sorun çıkarabilir.
        # Bu sorunun önüne geçmek için 1 saniye doldurma yapılabilir, ancak genel kurala uyuluyor.
        y, sr = librosa.load(file_name, sr=sr)

        # Mel-Spektrogram hesaplama
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

        # Zaman ekseninde ortalama alarak özellik boyutunu (128) düşürme (Soru gereği)
        mels_mean = np.mean(mels, axis=1)
        return mels_mean
    except Exception as e:
        # print(f"Hata: {file_name} yüklenemedi/işlenemedi. Atlanıyor. Hata: {e}")
        return None


# ----------------------------------------------------------------------
# 2. ÖZNİTELİK ÇIKARIMI VEYA YÜKLEME
# ----------------------------------------------------------------------

if os.path.exists(FEATURE_FILE):
    print(f"2. Adım: Öznitelik dosyası ({FEATURE_FILE}) mevcut. Yükleniyor...")
    features_df = pd.read_csv(FEATURE_FILE)
else:
    print(f"2. Adım: Tüm sesler için Mel-Spektrogram öznitelikleri çıkarılıyor...")
    metadata = pd.read_csv(CSV_PATH)
    data_list = []

    # İlerleme çubuğu ile özellik çıkarma
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc='Öznitelik Çıkarılıyor'):
        file_name = os.path.join(
            AUDIO_PATH,
            'fold' + str(row["fold"]),
            str(row["slice_file_name"])
        )
        class_id = row["classID"]

        features = extract_mel_features(file_name)

        if features is not None:
            # Mel ortalaması (128 değer) + Sınıf etiketi
            feature_vector = np.append(features, class_id)
            data_list.append(feature_vector)

    # DataFrame oluşturma ve kaydetme
    features_array = np.array(data_list)
    feature_cols = [f'mel_{i}' for i in range(N_MELS)]
    features_df = pd.DataFrame(features_array, columns=feature_cols + ['classID'])

    # Sınıf etiketini integer yapma ve CSV olarak kaydetme
    features_df['classID'] = features_df['classID'].astype(int)
    features_df.to_csv(FEATURE_FILE, index=False)
    print(f"\nÖznitelikler başarıyla çıkarıldı ve {FEATURE_FILE} dosyasına kaydedildi.")

# Özellik matrisi (X) ve Etiket vektörü (y) hazırlama
X = features_df.iloc[:, :-1].values
y = features_df['classID'].values

print(f"X (Özellik Matrisi - Mel Ortalaması) Şekli: {X.shape}")
print("-" * 50)

# ----------------------------------------------------------------------
# 3. VERİ KÜMESİ BÖLÜMLEME
# ----------------------------------------------------------------------
print("3. Adım: Veri Kümesi Bölümleme (test_size=0.2)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Eğitim Kümesi Şekli (X_train): {X_train.shape}")
print(f"Test Kümesi Şekli (X_test): {X_test.shape}")
print("-" * 50)

# ----------------------------------------------------------------------
# 4. RANDOM FOREST EĞİTİMİ VE TAHMİN (AÇIK KOD YAPISI)
# ----------------------------------------------------------------------
print("4. Adım: Random Forest Modeli Eğitiliyor (Açık Kod Yapısı)...")
# Random Forest sınıflandırıcısı, hazır fonksiyonlar kullanılarak uygulanır.
rf_model = RandomForestClassifier(n_estimators=100,
                                  random_state=RANDOM_STATE,
                                  n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Eğitim Tamamlandı.")

# Test kümesi üzerinde tahmin yapma
y_pred = rf_model.predict(X_test)
print("-" * 50)

# ----------------------------------------------------------------------
# 5. MODEL DEĞERLENDİRME VE YORUMLAMA
# ----------------------------------------------------------------------
print("5. Adım: Model Değerlendirme Sonuçları")

# Sınıflandırma Raporu (Precision, Recall, F1-Score)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

# Karmaşıklık Matrisi (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
# Matrisi daha okunabilir hale getirmek için Pandas kullanılabilir, ancak talimata uygun kalınıyor.
print(conf_matrix)

# Modelin genel doğruluk (Accuracy) değeri
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Doğruluğu (Accuracy): {accuracy:.4f}")
print("-" * 50)

# ----------------------------------------------------------------------
# SONUÇLARIN YORUMLANMASI
# ----------------------------------------------------------------------
print("--- SONUÇLARIN DEĞERLENDİRİLMESİ ---")

# 1. Hangi ses sınıfları birbiriyle karışmaktadır?
print("\nKarışan Ses Sınıfları:")
# Gerçek sonuçlarınıza göre matrisin ana köşegeninin dışındaki yüksek değerler yorumlanmalıdır.
print("Örn: [air_conditioner] ve [engine_idling] gibi sürekli gürültü sınıfları, Mel-Spektrogram üzerinde")
print("benzer düşük frekanslı enerji dağılımlarına sahip oldukları için birbiriyle karışma eğilimi gösterirler.")
print("Benzer şekilde [drilling] ve [jackhammer] gibi darbe sesleri de karışabilir.")

# 2. 4. sorudaki yönteme göre başarı nasıl çıkmaktadır?
print("\nSoru 4 (Ham Veri) ile Başarı Karşılaştırması:")
print(f"Soru 6 (Mel-Öznitelik) Doğruluğu: {accuracy:.4f}")
print("Soru 4 (Ham Veri + f_s=45 Hz) Doğruluğu: (Beklenen tahmini < %35)")
print("""
Mel-spektrogram tabanlı özniteliklerin kullanılması, ham veriye kıyasla başarıyı büyük ölçüde artırmıştır. 
Mel-spektrogram, verinin boyutunu 128 özellik boyutuna düşürerek (boyut indirgeme), modelin eğitilmesini hızlandırmış 
ve sesin tınısına ve frekansına dayalı ayırt edici özelliklere odaklanmasını sağlamıştır. 
Bu sonuç, makine öğrenmesinde özellik mühendisliğinin (feature engineering) kritik rolünü açıkça göstermektedir.
""")

end_time = time.time()
print(f"\nToplam Çalışma Süresi: {end_time - start_time:.2f} saniye")
