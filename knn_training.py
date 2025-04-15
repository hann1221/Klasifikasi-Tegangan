import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# 1. Baca data
df = pd.read_csv("TRAINING DATA.csv")

# 2. Buat label klasifikasi dari nilai V2
def klasifikasi_tekanan(v):
    if v < 0.95:
        return "Bahaya"
    elif v <= 1.05:
        return "Normal"
    else:
        return "Overvoltage"

df["status_tegangan"] = df["V2"].apply(klasifikasi_tekanan)

# 3. Pisah fitur dan target untuk regresi & klasifikasi
X = df[['S', 'G', 'PL_2', 'PL_3', 'Q2', 'Q3', 'DELTA_2', 'DELTA_3']]
y_regresi = df["V2"]
y_klasifikasi = df["status_tegangan"]

# 4. Split data
X_train, X_test, y_reg_train, y_reg_test, y_klas_train, y_klas_test = train_test_split(
    X, y_regresi, y_klasifikasi, test_size=0.2, random_state=42
)

# 5. Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Training model regresi dan klasifikasi
model_regresi = KNeighborsRegressor(n_neighbors=3)
model_regresi.fit(X_train_scaled, y_reg_train)

model_klasifikasi = KNeighborsClassifier(n_neighbors=3)
model_klasifikasi.fit(X_train_scaled, y_klas_train)

# 7. Prediksi data baru (1 baris contoh)
data_baru = [[13, 600, 1.2, 1.5, 0.1, 0.08, -0.9, 0.3]]
data_baru_scaled = scaler.transform(data_baru)

v2_pred = model_regresi.predict(data_baru_scaled)[0]
status_pred = model_klasifikasi.predict(data_baru_scaled)[0]

# 8. Cetak hasil
print("===== HASIL PREDIKSI HYBRID =====")
print(f"Prediksi Nilai Tegangan (V2): {v2_pred:.3f} V")
print(f"Status Tegangan (KNN Klasifikasi): {status_pred}")

# 9. Validasi status berdasarkan hasil regresi
def status_dari_v2(v):
    if v < 0.95:
        return "Bahaya"
    elif v <= 1.05:
        return "Normal"
    else:
        return "Overvoltage"

status_dari_regresi = status_dari_v2(v2_pred)
print(f"Status Berdasarkan Regresi: {status_dari_regresi}")

# 10. Cek apakah prediksi klasifikasi konsisten dengan hasil regresi
if status_dari_regresi == status_pred:
    print("✅ Konsisten antara Regresi dan Klasifikasi")
else:
    print("⚠️  Tidak Konsisten! Perlu dicek lagi")
