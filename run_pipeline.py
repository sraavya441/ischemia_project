import pandas as pd
from pipeline import ClinicalECGPipeline
import os

def run_subset():
    print('Initializing pipeline...')
    pipeline = ClinicalECGPipeline(sampling_rate=250)
    filepath = 'ecg_dataset.csv'
    output_path = 'ml_ready_features.csv'
    
    print(f'Processing first 1000 rows of {filepath}...')
    label_map = {0: 'Transmural', 1: 'Subendocardial', 2: 'Normal'}
    severity_map = {0: 'High', 1: 'Moderate', 2: 'None'}
    
    chunk = next(pd.read_csv(filepath, chunksize=1000))
    processed_rows = []
    
    for i, (_, row) in enumerate(chunk.iterrows()):
        if i > 0 and i % 100 == 0:
            print(f'Processed {i}/1000 rows...')
            
        if 'target' in row.index:
            target = int(row['target'])
            signal = row.drop('target').values.astype(float)
        else:
            target = int(row.iloc[-1])
            signal = row.iloc[:-1].values.astype(float)
            
        clean_sig = pipeline.adaptive_denoise(signal)
        morphology_features = pipeline.extract_morphology(clean_sig)
        wavelet_features = pipeline.extract_wavelet_features(clean_sig)
        
        final_row = {}
        final_row.update(morphology_features)
        final_row.update(wavelet_features)
        final_row['label'] = label_map.get(target, 'Unknown')
        final_row['severity'] = severity_map.get(target, 'Unknown')
        processed_rows.append(final_row)
        
    df_chunk = pd.DataFrame(processed_rows)
    df_clean_chunk = pipeline.run_qa_checks(df_chunk)
    df_clean_chunk.to_csv(output_path, mode='w', header=True, index=False)
    print(f'Subset processing complete. Saved {len(df_clean_chunk)} clean rows to {output_path}.')
    
    try:
        print('Generating heatmap...')
        pipeline.generate_feature_heatmap(df_clean_chunk)
        print('Heatmap generated.')
    except Exception as e:
        print(f'Error generating heatmap: {e}')

if __name__ == '__main__':
    run_subset()
