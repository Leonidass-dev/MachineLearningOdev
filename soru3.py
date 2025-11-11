import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np

from google.colab import drive
drive.mount('/content/drive')
URBANSOUND_YOLU = '/content/drive/MyDrive/UrbanSound8K/'

ses_dosyalari = {
    "Köpek Sesi": f"{URBANSOUND_YOLU}fold5/86284-3-0-0.wav",
    "Korna Sesi ": f"{URBANSOUND_YOLU}fold1/24074-1-0-4.wav",
    "Silah Sesi ": f"{URBANSOUND_YOLU}fold1/7061-6-0-0.wav",
    "Karışık Kedi Sesi": f"{URBANSOUND_YOLU}fold10/22973-3-0-2.wav",
}

veri_seti = {}
for etiket, yol in ses_dosyalari.items():
    try:
        data, sr = librosa.load(yol, sr=44100)
        veri_seti[etiket] = {'y': data, 'sr': sr}
        print(f"{etiket} dosyası başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: {etiket} dosyası yüklenemedi. Yol: {yol}. Hata: {e}")
        veri_seti[etiket] = None # Hata durumunda boş bırak

for etiket, veri in veri_seti.items():
    if veri is None:
        continue # Yüklenemeyen dosyayı atla

    data = veri['y']
    sr = veri['sr']

    # 1. STFT (Spektrogram) hesaplama
    # D: Kısa Zamanlı Fourier Dönüşümü (Short-Time Fourier Transform)
    D = librosa.stft(data)
    # S_db: Ses enerjisinin daha iyi gözlemlenmesi için genliği dB (desibel) ölçeğine dönüştürme
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 2. Grafik (Figure) Oluşturma ve Çizdirme
    # Her dosya için yeni bir figür başlatılır
    plt.figure(figsize=(12, 6))

    # Spektrogramı çizdirme
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')

    # Renk çubuğu (colorbar) ekleme
    plt.colorbar(format='%+2.0f dB')

    # Başlık ve etiketler
    plt.title(f'Zaman-Frekans Grafiği (Spektrogram) - Ses: {etiket}')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Frekans (Hz)')
    plt.show() # Grafiği göster

    # 3. Ses dosyasını çalma kodu
    print(f"\n--- Ses: {etiket} (Oynatıcı) ---")
    display(Audio(data=data, rate=sr)) # Colab/Jupyter'da ses oynatma
    print("="*40) # Ayırıcı
