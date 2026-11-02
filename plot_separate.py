import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        # Load raw data
        raw_df = pd.read_csv('recorded_ecg_data.csv', header=None)
        raw_signal = raw_df.iloc[:, 0].values
        
        # Load denoised data
        denoised_df = pd.read_csv('denoised_ecg_data.csv')
        denoised_signal = denoised_df['Denoised_Amplitude'].values
        
        # Create separate plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(raw_signal, color='gray')
        ax1.set_title("Raw Noisy Signal (ESP32)")
        ax1.set_ylabel("Amplitude")
        
        ax2.plot(denoised_signal, color='#0044ff')
        ax2.set_title("Denoised Clinical Signal")
        ax2.set_xlabel("Samples (at 250 Hz)")
        ax2.set_ylabel("Amplitude")
        
        plt.tight_layout()
        
        plot_filename = 'separated_plots.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved separated plots to '{plot_filename}'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
