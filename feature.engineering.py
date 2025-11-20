# ============================================================
#  FEATURE ENGINEERING DATASET MIXUE (VERSI SESUAI TUGAS)
# ============================================================

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Tahap 0: Load dataset mentah
# ------------------------------------------------------------

df = pd.read_excel("datasetmixue.xlsx")
df = df[df['Hari'] != 'Hari'].reset_index(drop=True)

print("=== DATASET MENTAH ===")
print(df.head(), "\n")


# ------------------------------------------------------------
# Tahap CLEANING untuk Feature Engineering
# Mengubah struktur dataset (Anak-Anak, Remaja, Dewasa)
# ------------------------------------------------------------

clean_rows = []
current_group = None

for _, row in df.iterrows():
    hari = row['Hari']
    kal = str(row['Kalangan Umur'])

    # Jika baris kategori
    if pd.isna(hari):
        if "Anak" in kal:
            current_group = "Anak-Anak"
        elif "Remaja" in kal:
            current_group = "Remaja"
        elif "Dewasa" in kal:
            current_group = "Dewasa"
        continue
    
    clean_rows.append({
        "Hari": int(hari),
        "Kelompok": current_group,
        "Jumlah Orang": row["Kalangan Umur"],
        "Minuman": row["Menu"],
        "Ice Cream": row["Unnamed: 3"]
    })

df_clean = pd.DataFrame(clean_rows)

print("=== DATASET SETELAH CLEANING STRUKTUR  ===")
print(df_clean.head(), "\n")


# ------------------------------------------------------------
# Tahap 1: FEATURE CLEANING
# Membersihkan teks menu dan menghilangkan typo
# ------------------------------------------------------------

def extract_items(text):
    if pd.isna(text):
        return {}
    items = {}
    parts = str(text).split(",")
    for p in parts:
        p = p.strip()
        match = re.match(r"(\d+)\s+(.*)", p)
        if match:
            qty = int(match.group(1))
            name = match.group(2).strip().title()
            # perbaikan typo paling sering
            name = name.replace("Sundaae", "Sundae")
            name = name.replace("Lemoned", "Lemon")
            items[name] = items.get(name, 0) + qty
    return items

print("=== CONTOH FEATURE CLEANING MENU  ===")
print(extract_items("1 Boba Sundaae, 2 Ice Cream Corn"), "\n")


# ------------------------------------------------------------
# Tahap 2: FEATURE CREATION
# Ekstraksi menu menjadi kolom numerik + Total Penjualan
# ------------------------------------------------------------

all_items = []

for _, row in df_clean.iterrows():
    minum = extract_items(row["Minuman"])
    ice = extract_items(row["Ice Cream"])
    combined = {**minum, **ice}
    all_items.append(combined)

df_items = pd.DataFrame(all_items).fillna(0)

df_final = pd.concat([df_clean[["Hari", "Kelompok", "Jumlah Orang"]], df_items], axis=1)

df_final["Total Penjualan"] = df_items.sum(axis=1)

print("=== HASIL FEATURE CREATION ===")
print(df_final.head(), "\n")


# ------------------------------------------------------------
# Tahap 3: FEATURE ENCODING
# Mengubah kategori Kelompok menjadi angka
# ------------------------------------------------------------

mapping = {"Anak-Anak": 0, "Remaja": 1, "Dewasa": 2}
df_final["Kelompok_Code"] = df_final["Kelompok"].map(mapping)

print("=== HASIL FEATURE ENCODING ===")
print(df_final[["Kelompok", "Kelompok_Code"]].head(), "\n")


# ------------------------------------------------------------
# Tahap 4: FEATURE SCALING
# Normalisasi nilai menu menggunakan MinMaxScaler
# ------------------------------------------------------------

menu_cols = df_items.columns

scaler = MinMaxScaler()
df_scaled = df_final.copy()
df_scaled[menu_cols] = scaler.fit_transform(df_final[menu_cols])

print("=== HASIL FEATURE SCALING ===")
print(df_scaled.head(), "\n")


# ------------------------------------------------------------
# Dataset final siap upload ke GitHub
# ------------------------------------------------------------
df_scaled.to_csv("dataset_feature_engineered.csv", index=False)

print("FILE dataset_feature_engineered.csv berhasil dibuat!")
