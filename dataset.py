import numpy as np
import pandas as pd
import os

# --------------------------
# Parameters
# --------------------------
n_samples = 60000
output_dir = "realistic_multi_datasets"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Generate unique 6-digit IDs
# --------------------------
np.random.seed(42)
ids = np.random.choice(np.arange(100000, 999999), size=n_samples, replace=False)

# --------------------------
# Pain levels
# --------------------------
pain_score = np.random.normal(0, 1, n_samples) * 2 + 2
bins = [-np.inf, 0.5, 1.5, 2.5, 3.5, 4.5, np.inf]
labels = [0, 1, 2, 3, 4, 5]
pain_level = pd.cut(pain_score, bins=bins, labels=labels).astype(int)

pain_labels_map = {
    0: "No Pain",
    1: "Mild Pain",
    2: "Moderate Pain",
    3: "Severe Pain",
    4: "Very Severe Pain",
    5: "Worst Pain Possible"
}

# --------------------------
# Global features
# --------------------------
ages = np.random.randint(18, 80, n_samples)
bmi = np.round(np.random.normal(24, 4, n_samples), 1)
timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="min")
room_temp = np.round(np.random.normal(22, 2, n_samples), 1)

# --------------------------
# Noise function
# --------------------------
def add_noise(signal, scale=0.05):
    noise = np.random.normal(0, scale*np.std(signal), signal.shape)
    return signal + noise

# --------------------------
# ECG Dataset
# --------------------------
ecg = add_noise(np.random.normal(75, 12, n_samples), scale=0.1)
hrv_sdnn = add_noise(np.random.normal(50, 10, n_samples), scale=0.1)
hrv_rmssd = add_noise(np.random.normal(30, 5, n_samples), scale=0.1)

df_ecg = pd.DataFrame({
    "id": ids,
    "timestamp": timestamps,
    "ecg": ecg,
    "hrv_sdnn": hrv_sdnn,
    "hrv_rmssd": hrv_rmssd,
    "age": ages,
    "bmi": bmi,
    "pain_level": pain_level
})
df_ecg["pain_label"] = df_ecg["pain_level"].map(pain_labels_map)
df_ecg.to_csv(f"{output_dir}/ecg_dataset.csv", index=False)

# --------------------------
# PPG Dataset
# --------------------------
ppg = add_noise(np.random.normal(0.8, 0.2, n_samples), scale=0.1)
pulse_rate = add_noise(ecg + np.random.normal(0,3,n_samples), scale=0.05)
spo2 = add_noise(np.random.normal(98,1,n_samples), scale=0.01)

df_ppg = pd.DataFrame({
    "id": ids,
    "timestamp": timestamps,
    "ppg": ppg,
    "pulse_rate": pulse_rate,
    "spo2": spo2,
    "room_temp": room_temp,
    "pain_level": pain_level
})
df_ppg["pain_label"] = df_ppg["pain_level"].map(pain_labels_map)
df_ppg.to_csv(f"{output_dir}/ppg_dataset.csv", index=False)

# --------------------------
# GSR Dataset
# --------------------------
gsr = add_noise(np.random.normal(6,3,n_samples), scale=0.2)
skin_temp = add_noise(np.random.normal(33,1,n_samples), scale=0.05)

df_gsr = pd.DataFrame({
    "id": ids,
    "timestamp": timestamps,
    "gsr": gsr,
    "skin_temp": skin_temp,
    "pain_level": pain_level
})
df_gsr["pain_label"] = df_gsr["pain_level"].map(pain_labels_map)
df_gsr.to_csv(f"{output_dir}/gsr_dataset.csv", index=False)

# --------------------------
# EEG Dataset
# --------------------------
eeg = add_noise(np.random.normal(50,20,n_samples), scale=0.15)
alpha_power = add_noise(np.random.normal(10,2,n_samples), scale=0.1)
beta_power = add_noise(np.random.normal(12,3,n_samples), scale=0.1)
df_eeg = pd.DataFrame({
    "id": ids,
    "timestamp": timestamps,
    "eeg": eeg,
    "alpha_power": alpha_power,
    "beta_power": beta_power,
    "pain_level": pain_level
})
df_eeg["pain_label"] = df_eeg["pain_level"].map(pain_labels_map)
df_eeg.to_csv(f"{output_dir}/eeg_dataset.csv", index=False)

print("✅ All 4 datasets generated with only necessary columns!")
