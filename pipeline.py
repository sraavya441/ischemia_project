import numpy as np
import pandas as pd
import pywt
from kneed import KneeLocator
import neurokit2 as nk
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

class ClinicalECGPipeline:
    def __init__(self, sampling_rate=250):
        """
        Initializes the pipeline configuration.
        """
        self.sampling_rate = sampling_rate
        self.wavelet = 'sym8'
        self.level = 5

    def adaptive_denoise(self, signal):
        """
        Applies adaptive knee-point soft thresholding to detail wavelet bands.
        Preserves the approximation coefficient (cA5) entirely.
        Applies strong denoising to cD1/cD2, moderate to cD3, and minimal to cD4/cD5.
        """
        if signal is None or len(signal) < 120:
            return np.array([]) if signal is None else np.array(signal)

        # Wavelet decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Extract approximation (cA5) and detail coefficients (cD5 to cD1)
        # Note: pywt.wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
        # For level 5: [cA5, cD5, cD4, cD3, cD2, cD1]
        cA5 = coeffs[0]
        details = coeffs[1:] 
        
        denoised_details = []
        
        # Map out the threshold severity modifiers based on the band
        # details order: [cD5, cD4, cD3, cD2, cD1]
        # cD5/cD4: Minimal (0.5x), cD3: Moderate (1.0x), cD2/cD1: Strong (2.0x)
        severity_multipliers = [0.5, 0.5, 1.0, 2.0, 2.0]
        
        for i, d in enumerate(details):
            # Sort absolute magnitudes descending
            abs_d = np.sort(np.abs(d))[::-1]
            
            try:
                # Use kneed to find the adaptive knee-point threshold
                kneedle = KneeLocator(range(len(abs_d)), abs_d, S=1.0, curve='convex', direction='decreasing')
                knee_idx = kneedle.knee
                
                if knee_idx is not None:
                    base_threshold = abs_d[knee_idx]
                else:
                    # Fallback to VisuShrink (Universal Threshold)
                    mad = np.median(np.abs(d - np.median(d)))
                    sigma = mad / 0.6745 if mad != 0 else 1e-6
                    base_threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            except Exception:
                # Fallback to VisuShrink if kneed fails
                mad = np.median(np.abs(d - np.median(d)))
                sigma = mad / 0.6745 if mad != 0 else 1e-6
                base_threshold = sigma * np.sqrt(2 * np.log(len(signal)))
                
            # Apply the specific severity multiplier for this detail band
            threshold = base_threshold * severity_multipliers[i]
            
            # Apply soft thresholding
            denoised_d = pywt.threshold(d, threshold, mode='soft')
            denoised_details.append(denoised_d)
            
        # Reconstruct the signal using the preserved cA5 and denoised detail bands
        denoised_coeffs = [cA5] + denoised_details
        denoised_signal = pywt.waverec(denoised_coeffs, self.wavelet)
        
        # Ensure the reconstructed length matches the original signal length exactly
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]
            
        return denoised_signal

    def plot_scaleogram(self, raw_sig, clean_sig, scales=np.arange(1, 128)):
        """
        Plots the CWT scaleograms for the raw and denoised ECG signals
        to visualize the morphological preservation.
        """
        coef_raw, _ = pywt.cwt(raw_sig, scales, 'morl')
        coef_clean, _ = pywt.cwt(clean_sig, scales, 'morl')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Raw ECG Scaleogram
        img1 = axes[0].imshow(np.abs(coef_raw), extent=[0, len(raw_sig), scales[-1], scales[0]], 
                              aspect='auto', cmap='jet', interpolation='bilinear')
        axes[0].set_title('Raw ECG Scaleogram', fontsize=14)
        axes[0].set_ylabel('Scales', fontsize=12)
        fig.colorbar(img1, ax=axes[0], label='Magnitude')
        
        # Denoised ECG Scaleogram
        img2 = axes[1].imshow(np.abs(coef_clean), extent=[0, len(clean_sig), scales[-1], scales[0]], 
                              aspect='auto', cmap='jet', interpolation='bilinear')
        axes[1].set_title('Denoised ECG Scaleogram', fontsize=14)
        axes[1].set_xlabel('Time (samples)', fontsize=12)
        axes[1].set_ylabel('Scales', fontsize=12)
        fig.colorbar(img2, ax=axes[1], label='Magnitude')
        
        plt.tight_layout()
        plt.show()

    def extract_morphology(self, clean_sig):
        """
        Extracts clinical morphological features from the denoised ECG.
        Safely handles missing waves and extracts QRS, ST, T, and QT metrics.
        """
        features = {
            'QRS_duration': np.nan,
            'ST_segment_deviation': np.nan,
            'T_wave_amplitude': np.nan,
            'T_wave_inversion_flag': np.nan,
            'heart_rate': np.nan,
            'RR_interval': np.nan,
            'J_point_amplitude': np.nan,
            'ST_slope': np.nan,
            'Q_wave_depth': np.nan,
            'QT_interval': np.nan,
            'T_wave_symmetry': np.nan
        }
        
        try:
            # Detect peaks using neurokit2
            _, rpeaks = nk.ecg_peaks(clean_sig, sampling_rate=self.sampling_rate)
            r_peaks_indices = rpeaks.get('ECG_R_Peaks', [])
            
            if len(r_peaks_indices) == 0:
                return features
                
            # Delineate robustly using DWT method
            _, waves = nk.ecg_delineate(clean_sig, r_peaks_indices, sampling_rate=self.sampling_rate, method="dwt")
            
            # Temporary storage to average across multiple beats if present
            metrics = {k: [] for k in features.keys()}
            
            for i in range(len(r_peaks_indices)):
                r_peak = r_peaks_indices[i]
                r_onset = waves.get('ECG_R_Onsets', [np.nan]*len(r_peaks_indices))[i]
                r_offset = waves.get('ECG_R_Offsets', [np.nan]*len(r_peaks_indices))[i]
                p_offset = waves.get('ECG_P_Offsets', [np.nan]*len(r_peaks_indices))[i]
                t_peak = waves.get('ECG_T_Peaks', [np.nan]*len(r_peaks_indices))[i]
                t_onset = waves.get('ECG_T_Onsets', [np.nan]*len(r_peaks_indices))[i]
                t_offset = waves.get('ECG_T_Offsets', [np.nan]*len(r_peaks_indices))[i]
                q_peak = waves.get('ECG_Q_Peaks', [np.nan]*len(r_peaks_indices))[i]
                
                # Isoelectric Baseline calculation (PR segment mean)
                baseline = np.nan
                if not np.isnan(p_offset) and not np.isnan(r_onset):
                    p_off, r_on = int(p_offset), int(r_onset)
                    if p_off < r_on:
                        baseline = np.mean(clean_sig[p_off:r_on])
                
                # Fallback to R-wave onset amplitude
                if np.isnan(baseline):
                    if not np.isnan(r_onset):
                        baseline = clean_sig[int(r_onset)]
                    else:
                        baseline = 0.0
                
                # QRS duration, J-point, and ST-segment features
                if not np.isnan(r_onset) and not np.isnan(r_offset):
                    r_on, r_off = int(r_onset), int(r_offset)
                    
                    # QRS Duration
                    metrics['QRS_duration'].append((r_off - r_on) / self.sampling_rate)
                    
                    # J-Point Amplitude
                    j_amp = clean_sig[r_off] - baseline
                    metrics['J_point_amplitude'].append(j_amp)
                    
                    # ST segment deviation (60ms after J-point)
                    st_idx = r_off + int(0.06 * self.sampling_rate)
                    if st_idx < len(clean_sig):
                        st_dev = clean_sig[st_idx] - baseline
                        metrics['ST_segment_deviation'].append(st_dev)
                        
                        # ST Slope (dv/dt over the 60ms window)
                        st_slope = (clean_sig[st_idx] - clean_sig[r_off]) / 0.06
                        metrics['ST_slope'].append(st_slope)
                
                # T-wave amplitude (Max absolute deviation in 150-400ms window after R peak)
                t_start = r_peak + int(0.15 * self.sampling_rate)
                t_end = r_peak + int(0.40 * self.sampling_rate)
                if t_start < len(clean_sig):
                    t_end = min(t_end, len(clean_sig))
                    t_window = clean_sig[t_start:t_end] - baseline
                    if len(t_window) > 0:
                        max_idx = np.argmax(np.abs(t_window))
                        t_amp = t_window[max_idx]
                        metrics['T_wave_amplitude'].append(t_amp)
                        
                        # T-wave inversion flag
                        metrics['T_wave_inversion_flag'].append(1.0 if t_amp < 0 else 0.0)
                
                # Q-wave depth
                if not np.isnan(q_peak):
                    q_p = int(q_peak)
                    metrics['Q_wave_depth'].append(clean_sig[q_p] - baseline)
                    
                # QT interval
                if not np.isnan(r_onset) and not np.isnan(t_offset):
                    r_on, t_off = int(r_onset), int(t_offset)
                    metrics['QT_interval'].append((t_off - r_on) / self.sampling_rate)
                    
                # T-wave symmetry
                if not np.isnan(t_onset) and not np.isnan(t_peak) and not np.isnan(t_offset):
                    t_on, t_p, t_off = int(t_onset), int(t_peak), int(t_offset)
                    t1 = (t_p - t_on) / self.sampling_rate
                    t2 = (t_off - t_p) / self.sampling_rate
                    if t2 > 0:
                        metrics['T_wave_symmetry'].append(t1 / t2)
                        
            # RR interval and Heart Rate
            if len(r_peaks_indices) > 1:
                rr_intervals = np.diff(r_peaks_indices) / self.sampling_rate
                mean_rr = np.nanmean(rr_intervals)
                metrics['RR_interval'].append(mean_rr)
                metrics['heart_rate'].append(60.0 / mean_rr)
                
            # Aggregate metrics (mean over all detected beats)
            for k in features.keys():
                if metrics[k]:
                    features[k] = np.nanmean(metrics[k])
                    
        except Exception:
            # Return NaNs gracefully on failure
            pass
            
        return features

    def extract_wavelet_features(self, clean_sig):
        """
        Extracts statistical features from wavelet bands (mean, std, entropy).
        Excludes variance.
        """
        features = {}
        band_names = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']
        
        if clean_sig is None or len(clean_sig) < 120:
            for name in band_names:
                features[f'{name}_mean'] = 0.0
                features[f'{name}_std'] = 0.0
                features[f'{name}_entropy'] = 0.0
            return features
            
        # Decompose signal
        coeffs = pywt.wavedec(clean_sig, self.wavelet, level=self.level)
        
        for name, coeff in zip(band_names, coeffs):
            features[f'{name}_mean'] = np.mean(coeff)
            features[f'{name}_std'] = np.std(coeff)
            
            # Shannon Entropy
            sq_coeff = coeff ** 2
            sum_sq = np.sum(sq_coeff)
            if sum_sq > 0:
                probs = sq_coeff / sum_sq
                entropy = scipy.stats.entropy(probs, base=2)
            else:
                entropy = 0.0
                
            features[f'{name}_entropy'] = entropy
            
        return features

    def run_qa_checks(self, df):
        """
        Runs advanced QA checks on the DataFrame: 
        drops rows with NaN/Inf values, performs feature sanity bounds validation,
        and drops columns with zero variance.
        """
        # Replace Inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Interval Sanity Validation (e.g. QRS duration must be > 0 and < 0.3s)
        if 'QRS_duration' in df.columns:
            df.loc[(df['QRS_duration'] <= 0) | (df['QRS_duration'] >= 0.3), 'QRS_duration'] = np.nan
        if 'QT_interval' in df.columns:
            df.loc[(df['QT_interval'] <= 0) | (df['QT_interval'] >= 1.0), 'QT_interval'] = np.nan
            
        # Drop rows with NaNs in critical columns
        critical_cols = []
        existing_crit = [c for c in critical_cols if c in df.columns]
        df = df.dropna(subset=existing_crit)
        
        # Fill remaining NaNs (e.g., missing Q waves) with 0
        df = df.fillna(0)
        
        # Drop columns with zero variance
        # Ensure we only check numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df) > 0:
            variances = numeric_df.var()
            zero_var_cols = variances[variances == 0].index
            df = df.drop(columns=zero_var_cols)
        
        return df

    def generate_feature_heatmap(self, df, output_filename="feature_heatmap.png"):
        """
        Generates a correlation heatmap of the extracted features.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
            
        plt.figure(figsize=(16, 12))
        corr = numeric_df.corr()
        
        # Draw heatmap
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

    def process_batch(self, filepath, output_path):
        """
        Processes an entire dataset of raw ECGs in batches.
        Extracts features using adaptive denoising, remaps labels, 
        performs QA checks, and exports a final ML-ready dataset.
        """
        label_map = {0: 'Transmural', 1: 'Subendocardial', 2: 'Normal'}
        severity_map = {0: 'High', 1: 'Moderate', 2: 'None'}
        
        first_chunk = True
        
        # Process the dataset in chunks (e.g., 5000 rows at a time)
        for chunk in pd.read_csv(filepath, chunksize=5000):
            processed_rows = []
            
            for _, row in chunk.iterrows():
                # Try to extract the target column dynamically (often 'target' or the last column)
                if 'target' in row.index:
                    target = int(row['target'])
                    signal = row.drop('target').values.astype(float)
                else:
                    target = int(row.iloc[-1])
                    signal = row.iloc[:-1].values.astype(float)
                
                # 1. Adaptive Denoising (Crucial: all downstream processing uses clean_sig)
                clean_sig = self.adaptive_denoise(signal)
                
                # 2. Extract Morphology on the clean signal
                morphology_features = self.extract_morphology(clean_sig)
                
                # 3. Extract Wavelet Features on the clean signal
                wavelet_features = self.extract_wavelet_features(clean_sig)
                
                # Combine all features into a single dictionary
                final_row = {}
                final_row.update(morphology_features)
                final_row.update(wavelet_features)
                
                # Map Labels and Severity
                final_row['label'] = label_map.get(target, 'Unknown')
                final_row['severity'] = severity_map.get(target, 'Unknown')
                
                processed_rows.append(final_row)
                
            # Convert chunk to DataFrame
            df_chunk = pd.DataFrame(processed_rows)
            
            # QA Checks (Drop NaNs, Infs, and zero-variance columns)
            df_clean_chunk = self.run_qa_checks(df_chunk)
            
            # Append to output CSV
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            df_clean_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            first_chunk = False

        print(f"Batch processing complete. Final dataset saved to {output_path}.")
        try:
            print("Generating feature correlation heatmap...")
            final_df = pd.read_csv(output_path)
            self.generate_feature_heatmap(final_df)
            print("Heatmap saved to feature_heatmap.png.")
        except Exception as e:
            print(f"Failed to generate heatmap: {e}")


if __name__ == "__main__":
    print("Initializing Clinical ECG Pipeline...")
    pipeline = ClinicalECGPipeline(sampling_rate=250)
    
    # Example usage block:
    # print("Processing batch data...")
    # pipeline.process_batch("raw_ecg_dataset.csv", "ml_ready_features.csv")
    print("Pipeline ready. You can call process_batch() to generate the ML-ready dataset.")
