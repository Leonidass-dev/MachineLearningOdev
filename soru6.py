# -*- coding: utf-8 -*-
"""
Makine Öğrenmesi - Soru 6
Sıfırdan Random Forest (Karar Ağaçları) uygulaması
Özellikler: Mel-Spektrogram (n_mels=128)
Amaç: from-scratch yaklaşımıyla eğitim + test ve metrik analizi
"""

import os, time, numpy as np, pandas as pd, librosa
from collections import Counter

# =========================================
# 1. Temel Ayarlar ve Dosya Kontrolleri
# =========================================
BASE_PATH = r"C:\Users\Leonidas\Downloads\archive"
META_CSV = os.path.join(BASE_PATH, "UrbanSound8K.csv")
assert os.path.exists(META_CSV), f"❌ Metadata bulunamadı: {META_CSV}"

FEATURE_CSV = "mel_features_rf.csv"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
rng = np.random.default_rng(42)

# =========================================
# 2. Mel-Spektrogram Öznitelik Çıkarımı
# =========================================
def mel_extract(path):
    """Bir ses dosyasından log-mel spektrum öznitelikleri çıkarır."""
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, fmax=SAMPLE_RATE/2
        )
        return np.log1p(np.mean(mel, axis=1)).astype(np.float32)
    except Exception as e:
        print(f"Hata ({os.path.basename(path)}): {e}")
        return None

if not os.path.exists(FEATURE_CSV):
    print("🔹 Özellikler çıkarılıyor (ilk kez)...")
    meta = pd.read_csv(META_CSV)
    feature_rows = []
    for i, row in meta.iterrows():
        wav = os.path.join(BASE_PATH, f"fold{row['fold']}", row['slice_file_name'])
        f = mel_extract(wav)
        if f is None:
            continue
        feature_dict = {f"m{i}": v for i, v in enumerate(f)}
        feature_dict["class_id"] = int(row["classID"])
        feature_rows.append(feature_dict)
        if (i + 1) % 400 == 0:
            print(f"  → {i+1}/{len(meta)} dosya işlendi...")
    pd.DataFrame(feature_rows).to_csv(FEATURE_CSV, index=False)
    print(f"✅ Özellik dosyası kaydedildi: {FEATURE_CSV}")
else:
    print(f"📁 Var olan özellik dosyası kullanılacak: {FEATURE_CSV}")

# =========================================
# 3. Veriyi Yükleme ve Bölme
# =========================================
df = pd.read_csv(FEATURE_CSV)
X = df.drop("class_id", axis=1).to_numpy(np.float32)
y = df["class_id"].to_numpy(np.int64)
labels = np.unique(y)
n_feats = X.shape[1]

print(f"Veri Yüklendi → X: {X.shape}, y: {y.shape}")

# Stratified bölme (test_size = 0.2)
train_idx, test_idx = [], []
for cls in labels:
    idx = np.where(y == cls)[0]
    rng.shuffle(idx)
    n_test = max(1, int(0.2 * len(idx)))
    test_idx.extend(idx[:n_test])
    train_idx.extend(idx[n_test:])
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Eğitim: {X_train.shape[0]} örnek | Test: {X_test.shape[0]} örnek")

# =========================================
# 4. Karar Ağacı ve Orman Tanımı
# =========================================
class LeafNode:
    """Yaprak düğüm: en sık görülen sınıfı saklar."""
    def __init__(self, value):
        self.value = value

class SplitNode:
    """İç düğüm: hangi özelliğe göre ayrılacağını tutar."""
    def __init__(self, feat, thr, left, right):
        self.feature = feat
        self.threshold = thr
        self.left = left
        self.right = right

class SimpleTree:
    """Basit Karar Ağacı (Gini kriterine göre bölünür)."""
    def __init__(self, max_depth=6, min_samples=8, n_subfeats=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_subfeats = n_subfeats
        self.root = None

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= self.min_samples:
            return LeafNode(Counter(y).most_common(1)[0][0])

        n_feats = int(np.sqrt(X.shape[1]))  # sadece sqrt(128)=11 özellik kullan
        feat_idx = rng.choice(X.shape[1], n_feats, replace=False)

        best_gain, best_feat, best_thr = 0, None, None
        for f in feat_idx:
            for t in np.unique(X[:, f]):
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if np.any(left_mask) and np.any(right_mask):
                    gain = self._gain(y, y[left_mask], y[right_mask])
                    if gain > best_gain:
                        best_gain, best_feat, best_thr = gain, f, t

        if best_feat is None:
            return LeafNode(Counter(y).most_common(1)[0][0])

        left = self.fit(X[X[:, best_feat] <= best_thr], y[X[:, best_feat] <= best_thr], depth + 1)
        right = self.fit(X[X[:, best_feat] > best_thr], y[X[:, best_feat] > best_thr], depth + 1)
        self.root = SplitNode(best_feat, best_thr, left, right)
        return self.root

    def _gain(self, parent, left, right):
        def gini(arr):
            p = np.bincount(arr) / len(arr)
            return 1 - np.sum(p * p)
        p = len(left) / len(parent)
        return gini(parent) - (p * gini(left) + (1 - p) * gini(right))

    def predict_one(self, x, node=None):
        node = node or self.root
        while isinstance(node, SplitNode):
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

class RandomForest:
    """Basit Random Forest (from scratch)."""
    def __init__(self, n_trees=15, max_depth=7, min_samples=8, sample_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        start_total = time.time()
        for i in range(self.n_trees):
            bag_size = int(self.sample_ratio * n)
            idx = rng.choice(n, bag_size, replace=True)
            X_s, y_s = X[idx], y[idx]

            tree = SimpleTree(max_depth=self.max_depth, min_samples=self.min_samples)
            start_tree = time.time()
            tree.fit(X_s, y_s)
            self.trees.append(tree)

            elapsed_tree = time.time() - start_tree
            progress = ((i + 1) / self.n_trees) * 100
            total_min = (time.time() - start_total) / 60
            print(f"[{i+1}/{self.n_trees}] 🌲 Ağaç tamamlandı ({elapsed_tree:.1f} sn, %{progress:.0f}) | Toplam: {total_min:.1f} dk")

        print(f"\n🌳 Eğitim tamamlandı ({(time.time() - start_total)/60:.1f} dk)")

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        final = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final)

# =========================================
# 5. Model Eğitimi ve Değerlendirme
# =========================================
print("\n🧠 Model eğitimi başlatılıyor...")
forest = RandomForest(n_trees=75, max_depth=10, min_samples=10, sample_ratio=0.8)
forest.fit(X_train, y_train)

print("✅ Eğitim tamamlandı, test verisinde tahmin yapılıyor...")
y_pred = forest.predict(X_test)

# =========================================
# 6. Metrik Hesaplama (Basit versiyon)
# =========================================
def confusion(y_true, y_pred, k=None):
    k = k or int(max(y_true.max(), y_pred.max()) + 1)
    mat = np.zeros((k, k), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        mat[yt, yp] += 1
    return mat

cm = confusion(y_test, y_pred, k=len(labels))
acc = (y_pred == y_test).mean()
prec = np.diag(cm) / (np.sum(cm, axis=0) + 1e-9)
rec = np.diag(cm) / (np.sum(cm, axis=1) + 1e-9)

print("\n===== SONUÇLAR =====")
print(f"Accuracy: {acc:.3f}")
print(f"Precision (macro): {np.mean(prec):.3f}")
print(f"Recall (macro): {np.mean(rec):.3f}")
print("Confusion Matrix:\n", cm)

# =========================================
# 7. Karışan Sınıflar
# =========================================
meta_full = pd.read_csv(META_CSV)
id2name = dict(zip(meta_full["classID"], meta_full["class"]))

print("\n--- En Çok Karışan Sınıflar ---")
for c in labels:
    row = cm[c].copy()
    row[c] = 0
    if np.sum(row) == 0:
        continue
    most_conf = np.argmax(row)
    print(f"{id2name[c]:<18} ↔ {id2name[most_conf]:<18} ({row[most_conf]} hata)")

print("\n✅ Bitti – Random Forest (özgün sürüm, ilerlemeli)")
