# ============================================
# TRAINING MODEL FAKE NEWS DETECTION
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("📖 Membaca dataset...")

# Cek apakah folder dataset ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print("⚠️  Folder 'dataset' tidak ditemukan!")
    print("📥 Silakan download dataset dari Kaggle dan masukkan Fake.csv & True.csv ke folder dataset/")
    exit()

fake_path = 'dataset/Fake.csv'
true_path = 'dataset/True.csv'

# Cek apakah file dataset ada
if not os.path.exists(fake_path):
    print(f"❌ File tidak ditemukan: {fake_path}")
    print("📥 Download dataset dari: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    exit()

if not os.path.exists(true_path):
    print(f"❌ File tidak ditemukan: {true_path}")
    exit()

# Baca dataset
try:
    fake_df = pd.read_csv(fake_path, encoding='utf-8')
    true_df = pd.read_csv(true_path, encoding='utf-8')
    print(f"✅ Fake.csv: {len(fake_df)} berita")
    print(f"✅ True.csv: {len(true_df)} berita")
except Exception as e:
    print(f"❌ Error membaca file: {e}")
    exit()

# Tambahkan label
fake_df['label'] = 0  # 0 = Fake
true_df['label'] = 1  # 1 = Real

# Gabungkan dataset
df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

# Gabungkan title dan text untuk analisis
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Hapus data duplikat
df = df.drop_duplicates(subset='content')

print(f"✅ Total data setelah cleaning: {len(df)} berita")

# Pisahkan fitur (X) dan target (y)
X = df['content']
y = df['label']

# Split data training 80% dan testing 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Melatih model dengan Naive Bayes...")

# Buat pipeline: TF-IDF Vectorizer + Naive Bayes
model = make_pipeline(
    TfidfVectorizer(max_features=5000, stop_words='english'),
    MultinomialNB(alpha=0.1)
)

# Training
model.fit(X_train, y_train)

# Evaluasi akurasi
accuracy = model.score(X_test, y_test)
print(f"✅ Akurasi model: {accuracy * 100:.2f}%")

# Simpan model
joblib.dump(model, 'model.pkl')
print("💾 Model disimpan sebagai model.pkl")

# Test contoh
test_texts = [
    "NASA successfully launched Artemis I mission to the moon today",  # Real news
    "Breaking: Bill Gates admits vaccines contain microchips for tracking"  # Fake news
]

print("\n🔍 Testing contoh:")
for text in test_texts:
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    confidence = max(prob) * 100
    result = "REAL" if pred == 1 else "FAKE"
    print(f"  Teks: {text[:50]}... → {result} ({confidence:.1f}% confidence)")

print("\n✅ Selesai! Jalankan 'python app.py' untuk membuka web app")