import pandas as pd
import numpy as np

def calculate_snr(clean_signal, noise):
    # Calculate power of the signal
    power_signal = np.sum(clean_signal ** 2)
    # Calculate power of the noise
    power_noise = np.sum(noise ** 2)
    
    # Calculate SNR in decibels (dB)
    snr_db = 10 * np.log10(power_signal / power_noise)
    return snr_db

def main():
    try:
        # Load raw data (noisy)
        raw_df = pd.read_csv('recorded_ecg_data.csv', header=None)
        raw_signal = raw_df.iloc[:, 0].values
        
        # Load denoised data
        denoised_df = pd.read_csv('denoised_ecg_data.csv')
        denoised_signal = denoised_df['Denoised_Amplitude'].values
        
        # Ensure they are the same length
        min_len = min(len(raw_signal), len(denoised_signal))
        raw_signal = raw_signal[:min_len]
        denoised_signal = denoised_signal[:min_len]
        
        # In our case, the "noise" is whatever the wavelet transform removed
        noise = raw_signal - denoised_signal
        
        # 1. Calculate SNR of the raw signal (before denoising)
        # Assuming the denoised signal is the "ground truth" for the calculation
        raw_snr = calculate_snr(denoised_signal, noise)
        
        print("\n--- CardioVision SNR Metrics ---")
        print("We treat the Wavelet Denoised signal as our 'Ground Truth'.")
        print(f"Signal-to-Noise Ratio (SNR) of Raw Hardware Signal: {raw_snr:.2f} dB")
        
        # Also let's show noise reduction
        variance_raw = np.var(raw_signal)
        variance_clean = np.var(denoised_signal)
        variance_noise = np.var(noise)
        
        print(f"\nRaw Signal Variance: {variance_raw:.2f}")
        print(f"Noise Variance Removed: {variance_noise:.2f}")
        
    except Exception as e:
        print(f"Error calculating SNR: {e}")

if __name__ == "__main__":
    main()
