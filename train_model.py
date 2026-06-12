import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

print("📖 Membaca dataset...")

if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print("⚠️ Folder 'dataset' tidak ditemukan!")
    exit()

fake_path = 'dataset/Fake.csv'
true_path = 'dataset/True.csv'

if not os.path.exists(fake_path):
    print(f"❌ File tidak ditemukan: {fake_path}")
    exit()

if not os.path.exists(true_path):
    print(f"❌ File tidak ditemukan: {true_path}")
    exit()

fake_df = pd.read_csv(fake_path, encoding='utf-8')
true_df = pd.read_csv(true_path, encoding='utf-8')

print(f"✅ Fake.csv: {len(fake_df)} berita")
print(f"✅ True.csv: {len(true_df)} berita")

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
df = df.drop_duplicates(subset='content')

print(f"✅ Total data setelah cleaning: {len(df)} berita")

X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Melatih model dengan Logistic Regression...")

model = make_pipeline(
    TfidfVectorizer(max_features=5000, stop_words='english'),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Akurasi model: {accuracy * 100:.2f}%")

joblib.dump(model, 'model.pkl')
print("💾 Model disimpan sebagai model.pkl")

print("\n✅ Selesai!")
