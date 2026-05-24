import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------
# Load datasets
# --------------------------
df_ecg = pd.read_csv("realistic_multi_datasets/ecg_dataset.csv")
df_ppg = pd.read_csv("realistic_multi_datasets/ppg_dataset.csv")
df_gsr = pd.read_csv("realistic_multi_datasets/gsr_dataset.csv")
df_eeg = pd.read_csv("realistic_multi_datasets/eeg_dataset.csv")

# Merge datasets on 'id' only
df_merged = df_ecg.merge(df_ppg, on='id', suffixes=('_ecg','_ppg')) \
                  .merge(df_gsr, on='id', suffixes=('','_gsr')) \
                  .merge(df_eeg, on='id', suffixes=('','_eeg'))

# --------------------------
# Drop non-numeric / unwanted columns
# --------------------------
exclude_cols = ['id','timestamp','pain_label','pain_level']
numeric_features = [c for c in df_merged.columns if c not in exclude_cols and np.issubdtype(df_merged[c].dtype, np.number)]

X = df_merged[numeric_features].fillna(0).values
y = pd.get_dummies(df_merged['pain_level']).values  # one-hot labels

# --------------------------
# Scale features
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler_all_features.pkl")

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for CNN-LSTM
X_train_dl = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --------------------------
# Build CNN-LSTM model
# --------------------------
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_dl.shape[1],1)),
    MaxPooling1D(pool_size=2),
    LSTM(50, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# Train model
# --------------------------
model.fit(X_train_dl, y_train, validation_data=(X_test_dl, y_test), epochs=25, batch_size=32, verbose=1)

# Save model
model.save("models/cnn_lstm_all_features.keras")
print("✅ Training complete. Model and scaler saved!")
