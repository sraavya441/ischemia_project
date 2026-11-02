import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
from streamlit_autorefresh import st_autorefresh
from pipeline import ClinicalECGPipeline

# Define LSTM Architecture
class ECGLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=3):
        super(ECGLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Page config
st.set_page_config(page_title="CardioVision Dashboard", page_icon="🫀", layout="wide")

# Custom CSS for modern styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫀 CardioVision ML Clinical Dashboard")
st.markdown("Real-time ECG Ischemia Classification & SHAP Explainability")

# --- Load Models & Pipeline ---
@st.cache_resource
def load_models():
    rf = joblib.load('rf_model.pkl')
    le = joblib.load('label_encoder.pkl')
    pipeline = ClinicalECGPipeline(sampling_rate=250)
    
    lstm_model = None
    lstm_le = None
    lstm_ready = False
    
    try:
        lstm_le = joblib.load('lstm_label_encoder.pkl')
        lstm_model = ECGLSTM(num_classes=len(lstm_le.classes_))
        # The old model is incompatible (not bidirectional), so this will fail until the new one finishes training
        lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location='cpu'))
        lstm_model.eval()
        lstm_ready = True
    except Exception as e:
        print(f"LSTM not ready yet: {e}")
        
    return rf, le, pipeline, lstm_model, lstm_le, lstm_ready

try:
    rf_model, label_encoder, ecg_pipeline, lstm_model, lstm_le, lstm_ready = load_models()
except Exception as e:
    st.error(f"Failed to load base models: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Data Input")
input_method = st.sidebar.radio("Choose Input Method:", ("Live Hardware Sync", "Demo Mode (Random Signal)", "Upload CSV"))

if 'current_input_method' not in st.session_state:
    st.session_state['current_input_method'] = input_method

if st.session_state['current_input_method'] != input_method:
    # Clear state when switching input methods
    for key in ['signal', 'analyzed_signal', 'analysis_results', 'demo_true_label', 'recorded_signal']:
        st.session_state.pop(key, None)
    st.session_state['needs_analysis'] = True
    st.session_state['analysis_complete'] = False
    st.session_state['current_input_method'] = input_method

signal = None

if input_method == "Live Hardware Sync":
    # Auto refresh only if connected or recording to prevent idle flashing
    if 'serial_pid' in st.session_state or st.session_state.get('is_recording', False):
        st_autorefresh(interval=2000, key="live_sync_refresh")
    st.sidebar.success("🟢 Live Sync Active: Monitoring for Bluetooth/USB data...")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Serial Acquisition Control**")
    
    # Dynamic COM port list using pyserial
    available_ports = []
    try:
        import serial.tools.list_ports
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        pass
        
    if available_ports:
        # Default to COM3 if present, else first available
        default_idx = available_ports.index("COM3") if "COM3" in available_ports else 0
        com_port = st.sidebar.selectbox("COM Port", available_ports, index=default_idx)
    else:
        st.sidebar.info("ℹ️ No active COM ports detected automatically.")
        com_port = st.sidebar.text_input("COM Port", "COM3")
        
    baud_rate = st.sidebar.selectbox("Baud Rate", [115200, 9600, 57600, 38400], index=0)
    
    if 'serial_pid' in st.session_state:
        status_text = "📡 Connecting..."
        if os.path.exists("serial_status.txt"):
            try:
                with open("serial_status.txt", "r") as f:
                    status_text = f.read().strip()
            except Exception:
                pass
        
        if "Failed" in status_text or "Error" in status_text:
            st.sidebar.error(status_text)
            # Clean up the pid since it failed
            try:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(st.session_state['serial_pid'])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
            del st.session_state['serial_pid']
        else:
            st.sidebar.success(status_text)
    else:
        st.sidebar.info("🔴 Serial Reader Stopped.")
    st.sidebar.markdown("---")
    
    if os.path.exists('latest_signal.json'):
        try:
            with open('latest_signal.json', 'r') as f:
                sig_data = json.load(f)
                
            if st.session_state.get('is_recording', False):
                # Set start index at the beginning of recording session
                if st.session_state.get('recording_start_idx') is None:
                    st.session_state['recording_start_idx'] = len(sig_data)
                
                start_idx = st.session_state['recording_start_idx']
                recorded_samples = sig_data[start_idx:]
                st.session_state['recorded_signal'] = recorded_samples
                signal = np.array(recorded_samples, dtype=float)
            else:
                signal = np.array(sig_data, dtype=float)
                
            st.session_state['signal'] = signal
            
            # The signal to analyze is the full recorded one if complete
            if st.session_state.get('analysis_complete', False):
                st.session_state['analyzed_signal'] = np.array(st.session_state.get('recorded_signal', sig_data), dtype=float)
            else:
                st.session_state['analyzed_signal'] = signal
                
            st.sidebar.info(f"Data loaded from latest sync ({len(sig_data)} total samples).")
        except Exception as e:
            st.sidebar.warning("Waiting for valid signal data...")
    else:
        st.sidebar.warning("No live data received yet. Start Serial Acquisition!")
        if 'signal' in st.session_state:
            del st.session_state['signal']
        if 'analyzed_signal' in st.session_state:
            del st.session_state['analyzed_signal']

elif input_method == "Demo Mode (Random Signal)":
    if st.sidebar.button("Generate Random ECG Signal"):
        try:
            df_raw = pd.read_csv('ecg_dataset_72k_balanced.csv', header=None)
            random_row = df_raw.sample(1).iloc[0]
            signal = random_row.iloc[:-1].values.astype(float)
            true_label_numeric = random_row.iloc[-1]
            label_map = {0.0: "Transmural", 1.0: "Subendocardial", 2.0: "Normal"}
            st.session_state['demo_true_label'] = label_map.get(float(true_label_numeric), "Unknown")
            st.session_state['signal'] = signal
            st.session_state['analyzed_signal'] = signal
            st.session_state['needs_analysis'] = True
            st.session_state['analysis_complete'] = True
        except Exception as e:
            st.sidebar.error(f"Failed to load dataset for demo: {e}")

elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Patient CSV Data", type=["csv"])
    if uploaded_file is not None:
        try:
            # Inspect first cell to check if a header row is present
            df_up = pd.read_csv(uploaded_file, header=None)
            
            has_header = False
            try:
                float(df_up.iloc[0, 0])
            except ValueError:
                has_header = True
                
            if has_header:
                uploaded_file.seek(0)
                df_up = pd.read_csv(uploaded_file)
                
            num_rows, num_cols = df_up.shape
            
            # If the CSV has very few columns (e.g. < 10), it is a continuous signal (time-series)
            if num_cols < 10:
                st.sidebar.info(f"Continuous signal detected: {num_rows} samples.")
                if num_cols == 1:
                    signal = pd.to_numeric(df_up.iloc[:, 0], errors='coerce').dropna().values.astype(float)
                else:
                    # Allow selecting which column holds the signal
                    col_names = [f"Col {i} ({df_up.columns[i]})" for i in range(num_cols)]
                    selected_col_idx = st.sidebar.selectbox("Select Signal Column", range(num_cols), format_func=lambda i: col_names[i])
                    signal = pd.to_numeric(df_up.iloc[:, selected_col_idx], errors='coerce').dropna().values.astype(float)
            else:
                # Treated as a dataset of multiple beats/patients where each row is a signal
                st.sidebar.info(f"Dataset detected: {num_rows} patients/beats (each row is a signal).")
                row_idx = st.sidebar.number_input("Select Patient/Beat Row Index", min_value=0, max_value=num_rows-1, value=0)
                row_data = df_up.iloc[row_idx]
                
                exclude_last = st.sidebar.checkbox("Exclude last column (True Label/Target)", value=True)
                if exclude_last:
                    signal = pd.to_numeric(row_data.iloc[:-1], errors='coerce').dropna().values.astype(float)
                    try:
                        lbl_val = row_data.iloc[-1]
                        st.sidebar.write(f"Row Label/Target: {lbl_val}")
                    except Exception:
                        pass
                else:
                    signal = pd.to_numeric(row_data, errors='coerce').dropna().values.astype(float)
                
            st.session_state['signal'] = signal
            st.session_state['analyzed_signal'] = signal
            st.session_state['needs_analysis'] = True
            st.session_state['analysis_complete'] = True
        except Exception as e:
            st.sidebar.error(f"Invalid CSV format: {e}")
    else:
        # Clear signal and related keys if no file is uploaded
        for key in ['signal', 'analyzed_signal', 'analysis_results', 'demo_true_label']:
            st.session_state.pop(key, None)

# --- Main Dashboard ---
if input_method == "Live Hardware Sync":
    st.markdown("### 🫀 Live Hardware Acquisition & Instructions")
    
    st.markdown("**Recording Settings**")
    timer_col1, timer_col2 = st.columns([1, 2])
    with timer_col1:
        duration_opt = st.selectbox("Recording Duration", ["10 seconds", "30 seconds", "1 minute", "Custom"])
    with timer_col2:
        if duration_opt == "10 seconds":
            duration_sec = 10
        elif duration_opt == "30 seconds":
            duration_sec = 30
        elif duration_opt == "1 minute":
            duration_sec = 60
        else:
            duration_sec = st.number_input("Custom Duration (seconds)", min_value=1, value=10)
            
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start", use_container_width=True, help="Connect hardware and display ECG"):
            if 'serial_pid' not in st.session_state:
                try:
                    with open("serial_status.txt", "w") as f:
                        f.write("📡 Connecting...")
                except Exception:
                    pass
                proc = subprocess.Popen([sys.executable, 'read_usb_serial.py', '--port', com_port, '--baud', str(baud_rate)])
                st.session_state['serial_pid'] = proc.pid
                st.success(f"Started connection on {com_port} at {baud_rate} baud!")
    with col2:
        is_rec = st.session_state.get('is_recording', False)
        if st.button("⏺️ Record", use_container_width=True, help="Start capturing the ECG", disabled=is_rec):
            st.session_state['is_recording'] = True
            st.session_state['recording_start_time'] = time.time()
            st.session_state['recording_duration'] = duration_sec
            st.session_state['recorded_signal'] = []
            st.session_state['recording_start_idx'] = None
            st.session_state['analysis_complete'] = False
            st.session_state['needs_analysis'] = False
            st.success("Recording started...")
    with col3:
        if st.button("⏹️ Stop", use_container_width=True, help="Stop the recording"):
            if 'serial_pid' in st.session_state:
                try:
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(st.session_state['serial_pid'])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
                del st.session_state['serial_pid']
            st.session_state['is_recording'] = False
            st.warning("Stopped recording and connection.")
            
    if st.session_state.get('is_recording', False):
        elapsed = time.time() - st.session_state.get('recording_start_time', time.time())
        duration = st.session_state.get('recording_duration', 10)
        remaining = max(0, duration - elapsed)
        
        st.markdown(f"**⏺️ Recording in Progress** - Elapsed: {elapsed:.1f}s / {duration}s | Remaining: {remaining:.1f}s")
        st.progress(min(elapsed / duration, 1.0))
        
        if elapsed >= duration:
            st.session_state['is_recording'] = False
            st.session_state['analysis_complete'] = True
            st.session_state['needs_analysis'] = True
            st.success("✅ Recording Complete! Analyzing data...")
            
            # Force final pull of the signal slice
            if os.path.exists('latest_signal.json'):
                try:
                    with open('latest_signal.json', 'r') as f:
                        sig_data = json.load(f)
                    start_idx = st.session_state.get('recording_start_idx')
                    if start_idx is not None:
                        st.session_state['recorded_signal'] = sig_data[start_idx:]
                except Exception:
                    pass
                    
            if 'recorded_signal' in st.session_state and len(st.session_state['recorded_signal']) > 0:
                pd.DataFrame(st.session_state['recorded_signal']).to_csv("recorded_ecg_data.csv", index=False, header=False)

    st.info("""
    **Instructions for the Collection of Electrodes:**
    1. Ask the subject to relax for **2 to 5 minutes** before taking the readings.
    2. **Electrode Placement**:
       - 🔴 **RA (Red)** → Right Arm
       - 🟡 **LL (Yellow)** → Left Leg (or lower left abdomen)
       - 🟢 **RL (Green)** → Right Leg (or lower right abdomen)
    3. A minimum of **10 seconds** of reading should be taken.
    """)
    st.markdown("---")

if 'signal' in st.session_state:
    signal = np.array(st.session_state['signal']).copy()
    analyzed_signal = np.array(st.session_state.get('analyzed_signal', signal)).copy()
    
    # Automatic Hardware Normalization (ESP32 outputs 0-4095 ADC. AI expects -1 to +2 mV distribution)
    if len(signal) > 0 and np.max(signal) > 10.0:
        sig_std = np.std(signal) if np.std(signal) != 0 else 1.0
        signal = ((signal - np.mean(signal)) / sig_std) * 0.39
        
    if len(analyzed_signal) > 0 and np.max(analyzed_signal) > 10.0:
        ana_std = np.std(analyzed_signal) if np.std(analyzed_signal) != 0 else 1.0
        analyzed_signal = ((analyzed_signal - np.mean(analyzed_signal)) / ana_std) * 0.39
    
    st.markdown("---")
    
    # Apply real-time denoising so the live monitor looks like a clean clinical ECG
    if input_method == "Live Hardware Sync":
        st.subheader("Live ECG Monitor")
        
        if signal is None or len(signal) == 0:
            st.info("📡 Waiting for hardware data stream. Click 'Start' to begin acquisition, then 'Record' to capture an ECG sample.")
        else:
            # Take the last 1000 samples for a clean scrolling live monitor visualization
            plot_signal = signal[-1000:] if len(signal) > 1000 else signal
            
            try:
                live_clean_sig = ecg_pipeline.adaptive_denoise(plot_signal)
            except Exception:
                live_clean_sig = plot_signal
                
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot raw signal very faintly in the background
            ax.plot(plot_signal, label="Raw Noise", color='gray', alpha=0.3)
            # Plot the clean clinical signal strongly
            ax.plot(live_clean_sig, label="Filtered ECG", color='#00ff00' if st.session_state.get('is_recording') else '#00ccff', linewidth=1.5)
            ax.set_title("Recording..." if st.session_state.get('is_recording') else "Live Feed (Filtered & Stabilized)")
            ax.set_facecolor('black')
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.legend()
            st.pyplot(fig)
        
    should_analyze = False
    if input_method != "Live Hardware Sync":
        should_analyze = True
    elif st.session_state.get('analysis_complete', False):
        should_analyze = True
        
    if should_analyze:
        if analyzed_signal is None or len(analyzed_signal) < 120:
            st.warning("⚠️ The captured signal is too short or empty for analysis (minimum 120 samples required). Please ensure your ESP32 hardware is connected, transmitting data, and the correct COM Port is selected.")
            st.session_state.pop('analysis_results', None)
        else:
            # Cache the analysis so it doesn't flicker/recompute every 2 seconds
            if st.session_state.get('needs_analysis', True):
                with st.spinner("Processing Full ECG Signal & AI Diagnosis..."):
                    clean_sig = ecg_pipeline.adaptive_denoise(analyzed_signal)
                
                # Isolate a single 300-sample beat for AI inference (matching training distribution)
                import neurokit2 as nk
                def get_ai_beat(sig):
                    if len(sig) <= 300: return sig
                    try:
                        _, rpeaks = nk.ecg_peaks(sig, sampling_rate=250)
                        r_idx = rpeaks.get('ECG_R_Peaks', [])
                        if len(r_idx) > 0:
                            center_r = r_idx[len(r_idx)//2]
                            start = max(0, center_r - 100)
                            end = start + 300
                            if end > len(sig):
                                end, start = len(sig), len(sig) - 300
                            return sig[start:end]
                    except: pass
                    return sig[:300]
                    
                ai_clean_sig = get_ai_beat(clean_sig)
                
                # Extract morphology from the FULL signal so we can accurately calculate Heart Rate across multiple beats!
                morphology = ecg_pipeline.extract_morphology(clean_sig)
                
                # Extract wavelets strictly from the ISOLATED beat for the AI models (to match training distributions)
                wavelets = ecg_pipeline.extract_wavelet_features(ai_clean_sig)
                
                # Combine and format features
                features = {}
                features.update(morphology)
                features.update(wavelets)
                df_features = pd.DataFrame([features]).fillna(0)
                expected_cols = getattr(rf_model, 'feature_names_in_', df_features.columns)
                df_features = df_features.reindex(columns=expected_cols, fill_value=0)
                
                # Engine 1: Random Forest
                rf_probs = rf_model.predict_proba(df_features)[0]
                prediction = rf_model.predict(df_features)
                diagnosis = label_encoder.inverse_transform(prediction)[0]
                
                # Engine 2: Deep Learning LSTM
                lstm_diagnosis = None
                lstm_probs = None
                if lstm_ready:
                    try:
                        # Convert flat signal array to tensor (batch=1, timesteps, features=1)
                        input_tensor = torch.FloatTensor(ai_clean_sig).unsqueeze(0).unsqueeze(-1)
                        with torch.no_grad():
                            lstm_out = lstm_model(input_tensor)
                            lstm_probs_raw = torch.nn.functional.softmax(lstm_out, dim=1).cpu().numpy()[0]
                        lstm_pred_idx = np.argmax(lstm_probs_raw)
                        lstm_val = lstm_le.inverse_transform([lstm_pred_idx])[0]
                        mapping = {0.0: "Transmural", 1.0: "Subendocardial", 2.0: "Normal"}
                        lstm_diagnosis = mapping.get(float(lstm_val), str(lstm_val))
                        
                        # Align LSTM probs to match RF ['Normal', 'Subendocardial', 'Transmural']
                        # LSTM raw is [Transmural(0), Subendocardial(1), Normal(2)]
                        lstm_probs = np.array([lstm_probs_raw[2], lstm_probs_raw[1], lstm_probs_raw[0]])
                    except Exception as e:
                        print("LSTM inference failed:", e)

                # Combined Soft Voting Output
                if lstm_ready and lstm_probs is not None:
                    combined_probs = (rf_probs + lstm_probs) / 2
                    combined_idx = np.argmax(combined_probs)
                    combined_diagnosis = label_encoder.classes_[combined_idx]
                    combined_confidence = combined_probs[combined_idx] * 100
                else:
                    combined_diagnosis = diagnosis
                    combined_confidence = np.max(rf_probs) * 100
                
                # Explain
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(df_features)
                if isinstance(shap_values, list):
                    pred_idx = prediction[0]
                    shap_vals = shap_values[pred_idx]
                else:
                    shap_vals = shap_values
                
                # Extract clean features dict for display
                def safe_fmt(val, dec, is_hr=False):
                    if pd.isna(val): 
                        return "N/A (Too short)" if is_hr else "N/A (Noisy Signal)"
                    return round(val, dec)
                    
                display_features = {
                    "QRS Duration (s)": safe_fmt(morphology.get('QRS_duration', np.nan), 3),
                    "Heart Rate (bpm)": safe_fmt(morphology.get('heart_rate', np.nan), 1, True),
                    "ST Deviation": safe_fmt(morphology.get('ST_segment_deviation', np.nan), 3),
                    "T-Wave Amp": safe_fmt(morphology.get('T_wave_amplitude', np.nan), 3)
                }
                
                # Cache results
                st.session_state['analysis_results'] = {
                    'clean_sig': clean_sig,
                    'diagnosis': diagnosis,
                    'lstm_diagnosis': lstm_diagnosis,
                    'combined_diagnosis': combined_diagnosis,
                    'combined_confidence': combined_confidence,
                    'display_features': display_features,
                    'df_features': df_features,
                    'shap_vals': shap_vals
                }
                st.session_state['needs_analysis'] = False

        res = st.session_state.get('analysis_results', {})
        if res:
            if 'demo_true_label' in st.session_state:
                st.info(f"🧪 **Demo Mode Ground Truth:** This sample is actually **{st.session_state['demo_true_label']}**")
                
            # 1. Dual-Engine Diagnosis Banner
            rf_diag = res['diagnosis']
            lstm_diag = res.get('lstm_diagnosis')
            combined_diag = res.get('combined_diagnosis', rf_diag)
            conf = res.get('combined_confidence', 0.0)
            
            color = "green" if combined_diag == "Normal" else "crimson" if combined_diag == "Transmural" else "darkorange"
            banner_html = f"""
            <div style="background-color:{color};padding:20px;border-radius:10px;color:white;text-align:center;box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <h3 style="margin:0;opacity:0.9;">CardioVision Dual-Engine AI Consensus</h3>
                <h1 style="margin:10px 0 0 0;font-size:42px;">{combined_diag}</h1>
                <p style="margin:5px 0 0 0;font-size:16px;opacity:0.9;">Combined Confidence: {conf:.1f}%</p>
                <div style="display:flex; justify-content:center; gap: 40px; margin-top:15px; background:rgba(0,0,0,0.1); border-radius:5px; padding:10px;">
                    <div>
                        <p style="margin:0;font-size:12px;opacity:0.7;">Engine 1 (Random Forest)</p>
                        <h4 style="margin:0;">{rf_diag}</h4>
                    </div>
                    <div>
                        <p style="margin:0;font-size:12px;opacity:0.7;">Engine 2 (Deep Learning LSTM)</p>
                        <h4 style="margin:0;">{lstm_diag if lstm_diag else 'Offline'}</h4>
                    </div>
                </div>
            </div>
            """
            st.markdown(banner_html + "<br>", unsafe_allow_html=True)
        
            # 2. Visualizations & Features
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Analyzed ECG Waveform")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(analyzed_signal, label="Raw Signal", alpha=0.5, color='gray')
                ax.plot(res['clean_sig'], label="Denoised (sym8)", color='blue', linewidth=1.5)
                ax.set_title("Adaptive Wavelet Denoising")
                ax.set_xlabel("Samples")
                ax.set_ylabel("Amplitude")
                ax.legend()
                st.pyplot(fig)
                
            with col2:
                st.subheader("Extracted Clinical Features")
                st.dataframe(pd.DataFrame(list(res['display_features'].items()), columns=["Feature", "Value"]), use_container_width=True)
                
            # 3. SHAP Explainability
            st.markdown("---")
            st.subheader("🤖 AI Explainability (SHAP)")
            st.markdown("This plot shows exactly which features influenced the AI's decision.")
            
            try:
                fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                shap.summary_plot(res['shap_vals'], res['df_features'], plot_type="bar", show=False)
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {e}")
        
else:
    st.info("👈 Please use the sidebar to generate or upload an ECG signal to begin.")
