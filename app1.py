import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Pain Level Prediction Dashboard", layout="wide")

# --------------------------
# Load Lottie Animation
# --------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_your_lottie_file.json")

# --------------------------
# Load model and scaler
# --------------------------
model = load_model("models/cnn_lstm_all_features.keras")
scaler = joblib.load("models/scaler_all_features.pkl")

# --------------------------
# Pain label mapping
# --------------------------
def map_pain_label(pain_level):
    if pain_level == 0:
        return "No Pain"
    elif pain_level <= 2:
        return "Mild"
    elif pain_level <= 4:
        return "Moderate"
    else:
        return "Severe"

# --------------------------
# Sidebar: Upload datasets
# --------------------------
st.sidebar.header("Upload CSV Datasets")
ecg_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
ppg_file = st.sidebar.file_uploader("Upload PPG CSV", type=["csv"])
gsr_file = st.sidebar.file_uploader("Upload GSR CSV", type=["csv"])
eeg_file = st.sidebar.file_uploader("Upload EEG CSV", type=["csv"])

df_merged = None
if ecg_file and ppg_file and gsr_file and eeg_file:
    df_ecg = pd.read_csv(ecg_file)
    df_ppg = pd.read_csv(ppg_file)
    df_gsr = pd.read_csv(gsr_file)
    df_eeg = pd.read_csv(eeg_file)

    df_merged = df_ecg.merge(df_ppg, on='id', suffixes=('_ecg','_ppg')) \
                      .merge(df_gsr, on='id', suffixes=('','_gsr')) \
                      .merge(df_eeg, on='id', suffixes=('','_eeg'))

    # FIX 1: Remove duplicate columns immediately after merge using loc + deduplicated index
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]  # Fix: drop duplicate columns right after merge

    st.sidebar.success("✅ Datasets merged successfully!")

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Prediction", "📈 Visualizations", "🔮 Forecasting"])

# ==========================
# TAB 1: HOME
# ==========================
with tab1:
    st.title("🩺 Pain Level Prediction Dashboard")
    if lottie_animation:
        from streamlit_lottie import st_lottie
        st_lottie(lottie_animation, height=200)
    st.markdown("""
    Welcome! This dashboard allows **pain level predictions** using physiological datasets (ECG, PPG, GSR, EEG)
    and visualizes the results.
    """)

# ==========================
# TAB 2: PREDICTION
# ==========================
with tab2:
    st.header("📊 Pain Level Prediction")

    if df_merged is None:
        st.warning("Please upload all datasets in the sidebar first.")
    else:
        st.subheader("Prediction")

        numeric_features = [c for c in df_merged.columns
                            if c not in ['id','timestamp','pain_level','pain_label']
                            and pd.api.types.is_numeric_dtype(df_merged[c])]
        manual_input = {}
        for feat in numeric_features:
            manual_input[feat] = st.number_input(feat, value=float(df_merged[feat].median()))

        if st.button("Predict Pain Level"):
            X_manual = np.array([list(manual_input.values())])
            if X_manual.shape[1] != scaler.mean_.shape[0]:
                st.error(f"Number of inputs ({X_manual.shape[1]}) does not match model features ({scaler.mean_.shape[0]}).")
            else:
                X_manual_scaled = scaler.transform(X_manual)
                X_manual_dl = X_manual_scaled.reshape(1, X_manual_scaled.shape[1], 1)
                pred_level = int(np.argmax(model.predict(X_manual_dl), axis=1)[0])
                pred_label = map_pain_label(pred_level)
                st.success(f"Predicted Pain Level: {pred_level}")
                st.info(f"Pain Label: {pred_label}")

                st.markdown("---")
                st.subheader("2️⃣ Dataset Prediction")
                exclude_cols = ['id','timestamp','pain_level','pain_label']
                X_pred_features = [c for c in df_merged.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_merged[c])]
                X_pred = df_merged[X_pred_features].fillna(0).values

                if X_pred.shape[1] != scaler.mean_.shape[0]:
                    st.error(f"Dataset feature count ({X_pred.shape[1]}) does not match model features ({scaler.mean_.shape[0]}).")
                else:
                    X_scaled = scaler.transform(X_pred)
                    X_dl = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                    df_merged['predicted_pain_level'] = np.argmax(model.predict(X_dl), axis=1)
                    df_merged['predicted_pain_label'] = df_merged['predicted_pain_level'].apply(map_pain_label)

                    # FIX 2: Deduplicate columns again after adding predicted columns to avoid downstream errors
                    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]  # Fix: re-deduplicate after adding prediction cols

                    st.success("✅ Predictions completed!")
                    st.dataframe(df_merged[['id','predicted_pain_level','predicted_pain_label']])

# ==========================
# TAB 3: VISUALIZATIONS
# ==========================
with tab3:
    st.header("📊 Visualizations")
    if df_merged is None or 'predicted_pain_level' not in df_merged.columns:
        st.warning("Please run predictions first in the Prediction tab.")
    else:
        df = df_merged.copy()

        # FIX 3: Deduplicate columns on the working copy before any operations
        df = df.loc[:, ~df.columns.duplicated()]  # Fix: ensures no duplicate columns exist in working copy

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df.index, y=df['predicted_pain_level'], mode='lines+markers'))
        fig_line.update_layout(title="Predicted Pain Level Over Samples", xaxis_title="Sample", yaxis_title="Pain Level")
        st.plotly_chart(fig_line, use_container_width=True)

        pain_counts = df['predicted_pain_label'].value_counts().reset_index()
        pain_counts.columns = ['label','count']
        fig_bar = px.bar(pain_counts, x='label', y='count', title="Pain Label Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Pie Chart: Pain Label Distribution")
        label_counts = df['predicted_pain_label'].value_counts()
        fig_pie = px.pie(values=label_counts.values, names=label_counts.index,
                         title="Predicted Pain Label Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("ECG vs Pain Level")
        if 'ecg' in df.columns and 'pain_level' in df.columns:
            fig3 = px.box(df, x='pain_level', y='ecg', points="all", title="ECG vs Pain Level")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Pulse Rate vs Pain Level")
        if 'pulse_rate' in df.columns and 'pain_level' in df.columns:
            fig4 = px.scatter(df, x='pain_level', y='pulse_rate', color='predicted_pain_level',
                              title="Pulse Rate vs Pain Level")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Skin Temp vs GSR")
        if 'skin_temp' in df.columns and 'gsr' in df.columns:
            fig5 = px.scatter(df, x='skin_temp', y='gsr', color='predicted_pain_level',
                              title="Skin Temp vs GSR Colored by Predicted Pain")
            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("EEG Alpha vs Beta Power")
        if 'alpha_power' in df.columns and 'beta_power' in df.columns:
            fig6 = px.scatter(df, x='alpha_power', y='beta_power', color='predicted_pain_level',
                              title="EEG Alpha vs Beta Power")
            st.plotly_chart(fig6, use_container_width=True)

        st.subheader("📊 Correlation Heatmap")

        exclude_cols = ['id', 'timestamp', 'pain_label', 'predicted_pain_label']  # FIX 4: removed 'pain_level' from exclude so it stays, added 'predicted_pain_label' (string col) to exclude

        # FIX 5: Build numeric_cols_corr WITHOUT manually appending 'predicted_pain_level' — it's already captured by is_numeric_dtype check
        numeric_cols_corr = [
            col for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]  # Fix: removed the manual append of 'predicted_pain_level' that caused the duplicate

        # FIX 6: Final dedup guard on the column list itself before computing correlation
        numeric_cols_corr = list(dict.fromkeys(numeric_cols_corr))  # Fix: remove any accidental duplicate strings in the list

        df_corr = df[numeric_cols_corr].corr()

        # FIX 7: Reset index so df_corr has no duplicate index/column names before passing to px.imshow
        df_corr = df_corr.loc[~df_corr.index.duplicated(), ~df_corr.columns.duplicated()]  # Fix: guard against duplicate labels in corr matrix

        fig_heatmap = px.imshow(df_corr, text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Key Features",
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ==========================
# TAB 4: FORECASTING
# ==========================
with tab4:
    st.header("🔮 Pain Forecasting")
    if df_merged is None or 'predicted_pain_level' not in df_merged.columns:
        st.warning("Please run predictions first in the Prediction tab.")
    else:
        df = df_merged.copy()

        # FIX 8: Deduplicate columns on forecasting working copy too
        df = df.loc[:, ~df.columns.duplicated()]  # Fix: prevent duplicate column errors in forecasting tab

        if 'timestamp' not in df.columns:
            st.error("Timestamp column is required for forecasting.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").dropna(subset=["timestamp"])

            df["smoothed_pain"] = df["predicted_pain_level"].rolling(window=9, min_periods=1).mean()
            df["rolling_forecast"] = df["smoothed_pain"].rolling(window=10, min_periods=1).mean()

            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            seq_len = 10
            scaler_forecast = MinMaxScaler()
            pain_scaled = scaler_forecast.fit_transform(df["smoothed_pain"].values.reshape(-1,1))

            X_seq, y_seq = [], []
            for i in range(len(pain_scaled)-seq_len):
                X_seq.append(pain_scaled[i:i+seq_len])
                y_seq.append(pain_scaled[i+seq_len])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

            # FIX 9: Guard against insufficient data for sequence modeling
            if len(X_seq) == 0:  # Fix: prevents crash when dataset is too small for seq_len
                st.error(f"Not enough data rows to build sequences. Need more than {seq_len} rows.")
            else:
                cnn_lstm_model = Sequential([
                    Conv1D(32, kernel_size=3, activation='relu', input_shape=(seq_len,1)),
                    MaxPooling1D(pool_size=2),
                    LSTM(50, activation='relu'),
                    Dense(1)
                ])
                cnn_lstm_model.compile(optimizer='adam', loss='mse')
                cnn_lstm_model.fit(X_seq, y_seq, epochs=25, batch_size=32, verbose=0)

                cnn_pred_scaled = cnn_lstm_model.predict(X_seq)
                cnn_pred = scaler_forecast.inverse_transform(cnn_pred_scaled).flatten()
                df["cnn_lstm_forecast"] = np.concatenate([np.full(seq_len, np.nan), cnn_pred])

                st.subheader("📈 Forecast Comparison")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["smoothed_pain"], mode="lines+markers", name="Smoothed Pain"))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rolling_forecast"], mode="lines+markers", name="Rolling Avg"))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cnn_lstm_forecast"], mode="lines+markers", name="CNN-LSTM Forecast"))
                fig.update_layout(title="Pain Forecast Comparison", xaxis_title="Timestamp", yaxis_title="Pain Level")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Forecasting Model Reports")

                def report(y_true, y_pred, name):
                    y_true_clean = y_true[~np.isnan(y_pred)]
                    y_pred_clean = y_pred[~np.isnan(y_pred)]
                    if len(y_true_clean) > 0:
                        mae = mean_absolute_error(y_true_clean, y_pred_clean)
                        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                        r2 = r2_score(y_true_clean, y_pred_clean)
                        st.write(f"**{name}** → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
                    else:
                        st.write(f"**{name}** → Not enough data to calculate metrics.")

                report(df["smoothed_pain"].values, df["rolling_forecast"].values, "Rolling Avg")
                report(df["smoothed_pain"].values, df["cnn_lstm_forecast"].values, "CNN-LSTM")














# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import joblib
# import requests
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# import plotly.graph_objects as go

# # --------------------------
# # Page config
# # --------------------------
# st.set_page_config(page_title="Pain Level Prediction Dashboard", layout="wide")

# # --------------------------
# # Load Lottie Animation
# # --------------------------
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_your_lottie_file.json")

# # --------------------------
# # Load model and scaler
# # --------------------------
# model = load_model("models/cnn_lstm_all_features.keras")
# scaler = joblib.load("models/scaler_all_features.pkl")

# # --------------------------
# # Pain label mapping
# # --------------------------
# def map_pain_label(pain_level):
#     if pain_level == 0:
#         return "No Pain"
#     elif pain_level <= 2:
#         return "Mild"
#     elif pain_level <= 4:
#         return "Moderate"
#     else:
#         return "Severe"

# # --------------------------
# # Sidebar: Upload datasets
# # --------------------------
# st.sidebar.header("Upload CSV Datasets")
# ecg_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
# ppg_file = st.sidebar.file_uploader("Upload PPG CSV", type=["csv"])
# gsr_file = st.sidebar.file_uploader("Upload GSR CSV", type=["csv"])
# eeg_file = st.sidebar.file_uploader("Upload EEG CSV", type=["csv"])

# df_merged = None
# if ecg_file and ppg_file and gsr_file and eeg_file:
#     df_ecg = pd.read_csv(ecg_file)
#     df_ppg = pd.read_csv(ppg_file)
#     df_gsr = pd.read_csv(gsr_file)
#     df_eeg = pd.read_csv(eeg_file)

#     # Merge datasets on 'id'
#     df_merged = df_ecg.merge(df_ppg, on='id', suffixes=('_ecg','_ppg')) \
#                       .merge(df_gsr, on='id', suffixes=('','_gsr')) \
#                       .merge(df_eeg, on='id', suffixes=('','_eeg'))
#     st.sidebar.success("✅ Datasets merged successfully!")

# # --------------------------
# # Tabs
# # --------------------------
# tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Prediction", "📈 Visualizations", "🔮 Forecasting"])

# # ==========================
# # TAB 1: HOME
# # ==========================
# with tab1:
#     st.title("🩺 Pain Level Prediction Dashboard")
#     if lottie_animation:
#         from streamlit_lottie import st_lottie
#         st_lottie(lottie_animation, height=200)
#     st.markdown("""
#     Welcome! This dashboard allows **pain level predictions** using physiological datasets (ECG, PPG, GSR, EEG)
#     and visualizes the results.
#     """)

# # ==========================
# # TAB 2: PREDICTION
# # ==========================
# with tab2:
#     st.header("📊 Pain Level Prediction")

#     if df_merged is None:
#         st.warning("Please upload all datasets in the sidebar first.")
#     else:
#         # --------------------------
#         # Manual input
#         # --------------------------
#         st.subheader("Prediction")

#         numeric_features = [c for c in df_merged.columns
#                             if c not in ['id','timestamp','pain_level','pain_label']
#                             and pd.api.types.is_numeric_dtype(df_merged[c])]
#         manual_input = {}
#         for feat in numeric_features:
#             manual_input[feat] = st.number_input(feat, value=float(df_merged[feat].median()))

#         if st.button("Predict Pain Level"):
#             # ----- Manual Prediction -----
#             X_manual = np.array([list(manual_input.values())])
#             if X_manual.shape[1] != scaler.mean_.shape[0]:
#                 st.error(f"Number of inputs ({X_manual.shape[1]}) does not match model features ({scaler.mean_.shape[0]}).")
#             else:
#                 X_manual_scaled = scaler.transform(X_manual)
#                 X_manual_dl = X_manual_scaled.reshape(1, X_manual_scaled.shape[1], 1)
#                 pred_level = int(np.argmax(model.predict(X_manual_dl), axis=1)[0])
#                 pred_label = map_pain_label(pred_level)
#                 st.success(f"Predicted Pain Level: {pred_level}")
#                 st.info(f"Pain Label: {pred_label}")

#                 st.markdown("---")
#                 # ----- Dataset Prediction -----
#                 st.subheader("2️⃣ Dataset Prediction")
#                 exclude_cols = ['id','timestamp','pain_level','pain_label']
#                 X_pred_features = [c for c in df_merged.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_merged[c])]
#                 X_pred = df_merged[X_pred_features].fillna(0).values

#                 if X_pred.shape[1] != scaler.mean_.shape[0]:
#                     st.error(f"Dataset feature count ({X_pred.shape[1]}) does not match model features ({scaler.mean_.shape[0]}).")
#                 else:
#                     X_scaled = scaler.transform(X_pred)
#                     X_dl = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
#                     df_merged['predicted_pain_level'] = np.argmax(model.predict(X_dl), axis=1)
#                     df_merged['predicted_pain_label'] = df_merged['predicted_pain_level'].apply(map_pain_label)
#                     st.success("✅ Predictions completed!")
#                     st.dataframe(df_merged[['id','predicted_pain_level','predicted_pain_label']])

# # ==========================
# # TAB 3: VISUALIZATIONS
# # ==========================
# with tab3:
#     st.header("📊 Visualizations")
#     if df_merged is None or 'predicted_pain_level' not in df_merged.columns:
#         st.warning("Please run predictions first in the Prediction tab.")
#     else:
#         df = df_merged.copy()
#         # Line plot
#         fig_line = go.Figure()
#         fig_line.add_trace(go.Scatter(x=df.index, y=df['predicted_pain_level'], mode='lines+markers'))
#         fig_line.update_layout(title="Predicted Pain Level Over Samples", xaxis_title="Sample", yaxis_title="Pain Level")
#         st.plotly_chart(fig_line, use_container_width=True)

#         # Bar plot
#         pain_counts = df['predicted_pain_label'].value_counts().reset_index()
#         pain_counts.columns = ['label','count']
#         fig_bar = px.bar(pain_counts, x='label', y='count', title="Pain Label Distribution")
#         st.plotly_chart(fig_bar, use_container_width=True)

#         st.subheader("Pie Chart: Pain Label Distribution")
#         label_counts = df['predicted_pain_label'].value_counts()
#         fig_pie = px.pie(values=label_counts.values, names=label_counts.index, 
#                          title="Predicted Pain Label Distribution")
#         st.plotly_chart(fig_pie, use_container_width=True)

#         st.subheader("ECG vs Pain Level")
#         if 'ecg' in df.columns and 'pain_level' in df.columns:
#             fig3 = px.box(df, x='pain_level', y='ecg', points="all", title="ECG vs Pain Level")
#             st.plotly_chart(fig3, use_container_width=True)

#         st.subheader("Pulse Rate vs Pain Level")
#         if 'pulse_rate' in df.columns and 'pain_level' in df.columns:
#             fig4 = px.scatter(df, x='pain_level', y='pulse_rate', color='predicted_pain_level',
#                               title="Pulse Rate vs Pain Level")
#             st.plotly_chart(fig4, use_container_width=True)

#         st.subheader("Skin Temp vs GSR")
#         if 'skin_temp' in df.columns and 'gsr' in df.columns:
#             fig5 = px.scatter(df, x='skin_temp', y='gsr', color='predicted_pain_level',
#                               title="Skin Temp vs GSR Colored by Predicted Pain")
#             st.plotly_chart(fig5, use_container_width=True)

#         st.subheader("EEG Alpha vs Beta Power")
#         if 'alpha_power' in df.columns and 'beta_power' in df.columns:
#             fig6 = px.scatter(df, x='alpha_power', y='beta_power', color='predicted_pain_level',
#                               title="EEG Alpha vs Beta Power")
#             st.plotly_chart(fig6, use_container_width=True)

#         st.subheader("📊 Correlation Heatmap")
#         df = df.loc[:, ~df.columns.duplicated()]
#         exclude_cols = ['id', 'timestamp', 'pain_label', 'pain_level']
#         numeric_cols_corr = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
#         if 'predicted_pain_level' in df.columns:
#             numeric_cols_corr.append('predicted_pain_level')

#         df_corr = df[numeric_cols_corr].corr()
#         fig_heatmap = px.imshow(df_corr, text_auto=True, aspect="auto",
#                         title="Correlation Heatmap of Key Features",
#                         color_continuous_scale='Viridis')
#         st.plotly_chart(fig_heatmap, use_container_width=True)

# # ==========================
# # TAB 4: FORECASTING
# # ==========================
# with tab4:
#     st.header("🔮 Pain Forecasting")
#     if df_merged is None or 'predicted_pain_level' not in df_merged.columns:
#         st.warning("Please run predictions first in the Prediction tab.")
#     else:
#         df = df_merged.copy()
#         if 'timestamp' not in df.columns:
#             st.error("Timestamp column is required for forecasting.")
#         else:
#             df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#             df = df.sort_values("timestamp").dropna(subset=["timestamp"])

#             # --------------------------
#             # Smooth predicted pain
#             # --------------------------
#             df["smoothed_pain"] = df["predicted_pain_level"].rolling(window=9, min_periods=1).mean()

#             # --------------------------
#             # Rolling Average Forecast
#             # --------------------------
#             df["rolling_forecast"] = df["smoothed_pain"].rolling(window=10, min_periods=1).mean()

#             # --------------------------
#             # CNN-LSTM Forecast
#             # --------------------------
#             from sklearn.preprocessing import MinMaxScaler
#             from tensorflow.keras.models import Sequential
#             from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
#             from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#             seq_len = 10
#             scaler_forecast = MinMaxScaler()
#             pain_scaled = scaler_forecast.fit_transform(df["smoothed_pain"].values.reshape(-1,1))

#             X_seq, y_seq = [], []
#             for i in range(len(pain_scaled)-seq_len):
#                 X_seq.append(pain_scaled[i:i+seq_len])
#                 y_seq.append(pain_scaled[i+seq_len])
#             X_seq, y_seq = np.array(X_seq), np.array(y_seq)

#             cnn_lstm_model = Sequential([
#                 Conv1D(32, kernel_size=3, activation='relu', input_shape=(seq_len,1)),
#                 MaxPooling1D(pool_size=2),
#                 LSTM(50, activation='relu'),
#                 Dense(1)
#             ])
#             cnn_lstm_model.compile(optimizer='adam', loss='mse')
#             cnn_lstm_model.fit(X_seq, y_seq, epochs=25, batch_size=32, verbose=0)

#             cnn_pred_scaled = cnn_lstm_model.predict(X_seq)
#             cnn_pred = scaler_forecast.inverse_transform(cnn_pred_scaled).flatten()
#             df["cnn_lstm_forecast"] = np.concatenate([np.full(seq_len, np.nan), cnn_pred])

#             # --------------------------
#             # Plot forecasts
#             # --------------------------
#             st.subheader("📈 Forecast Comparison")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=df["timestamp"], y=df["smoothed_pain"], mode="lines+markers", name="Smoothed Pain"))
#             fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rolling_forecast"], mode="lines+markers", name="Rolling Avg"))
#             fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cnn_lstm_forecast"], mode="lines+markers", name="CNN-LSTM Forecast"))
#             fig.update_layout(title="Pain Forecast Comparison", xaxis_title="Timestamp", yaxis_title="Pain Level")
#             st.plotly_chart(fig, use_container_width=True)

#             # --------------------------
#             # Forecasting Reports
#             # --------------------------
#             st.subheader("📊 Forecasting Model Reports")

#             def report(y_true, y_pred, name):
#                 y_true_clean = y_true[~np.isnan(y_pred)]
#                 y_pred_clean = y_pred[~np.isnan(y_pred)]
#                 if len(y_true_clean) > 0:
#                     mae = mean_absolute_error(y_true_clean, y_pred_clean)
#                     rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
#                     r2 = r2_score(y_true_clean, y_pred_clean)
#                     st.write(f"**{name}** → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
#                 else:
#                     st.write(f"**{name}** → Not enough data to calculate metrics.")

#             report(df["smoothed_pain"].values, df["rolling_forecast"].values, "Rolling Avg")
#             report(df["smoothed_pain"].values, df["cnn_lstm_forecast"].values, "CNN-LSTM")