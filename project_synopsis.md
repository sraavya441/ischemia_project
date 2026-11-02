# 🫀 CardioVision: Clinical ECG Diagnostic System

> [!NOTE]
> **Project Overview:** CardioVision is a comprehensive, clinical-grade diagnostic system designed to acquire, process, and analyze electrocardiogram (ECG) data in real-time. The primary objective is to provide high-precision classification of cardiac conditions (such as ischemia) by combining robust hardware data acquisition with a sophisticated **Dual-Engine AI** architecture.

---

## 🏗️ 1. High-Level System Architecture

The project seamlessly integrates hardware sensors, real-time data streaming, AI inference, and a clinical UI.

```mermaid
flowchart TD
    %% 1. Data Acquisition
    subgraph DA [1. Data Acquisition & Hardware]
        HW[ESP32 Microcontroller] -->|Analog ECG| Serial[Python Serial Reader]
        Serial -.->|JSON / TXT State| LocalState[(Local State Sync)]
    end

    %% 2. Signal Processing Pipeline
    subgraph SP [2. Signal Processing Pipeline]
        LocalState -->|Noisy ECG Data| DWT[DWT 'sym8' Decomposition]
        DWT -->|Extract cA5| cA5[Preserved Baseline Approximation]
        DWT -->|Extract Details| Details[cD1-cD5 Adaptive Soft Thresholding]
        cA5 --> Recon[Reconstruct Pristine Signal]
        Details --> Recon
        
        Recon --> Ext1[NeuroKit2 Morphological Features]
        Recon --> Ext2[Wavelet Statistical Features]
        Ext1 --> Tabular[(ML-Ready Tabular Data)]
        Ext2 --> Tabular
    end

    %% 3. Dual-Engine AI Architecture
    subgraph AI [3. Dual-Engine AI Consensus]
        Recon -->|Raw Time-Series| LSTM[Bidirectional LSTM Model]
        Tabular -->|Extracted Features| ML[Random Forest / XGBoost Models]
        
        LSTM --> PredLSTM[Deep Learning Prediction]
        ML --> PredML[Machine Learning Prediction]
        
        PredLSTM --> Consensus{Soft-Voting Consensus}
        PredML --> Consensus
    end

    %% 4. Clinical UI
    subgraph UI [4. Clinical User Interface]
        Consensus -- Agree --> HighConf[High Confidence Diagnosis]
        Consensus -- Disagree --> Alert[⚠️ AI Discrepancy Alert]
        
        HighConf --> Streamlit[Streamlit Clinical Dashboard]
        Alert --> Streamlit
        
        Streamlit -->|Visualizes| DenoisedTrace[Live Clean ECG Trace]
        Streamlit -->|Explains AI| SHAP[SHAP Feature Explainability]
    end

    %% Styles for readability
    classDef hardware fill:#2b3a42,stroke:#3f5765,stroke-width:2px,color:#fff
    classDef process fill:#1b4f72,stroke:#154360,stroke-width:2px,color:#fff
    classDef ai fill:#512e5f,stroke:#4a235a,stroke-width:2px,color:#fff
    classDef ui fill:#0e6251,stroke:#0b5345,stroke-width:2px,color:#fff
    
    class DA,HW,Serial,LocalState hardware
    class SP,DWT,cA5,Details,Recon,Ext1,Ext2,Tabular process
    class AI,LSTM,ML,PredLSTM,PredML,Consensus ai
    class UI,HighConf,Alert,Streamlit,DenoisedTrace,SHAP ui
```

---

## 📡 2. Signal Processing Pipeline (`pipeline.py`)

A critical component of CardioVision is transforming noisy clinical data into pristine signals while explicitly preserving baseline cardiovascular morphologies.

```mermaid
flowchart LR
    Raw[Raw Noisy ECG] --> DWT[DWT 'sym8' Decomposition]
    DWT --> cA5[Preserve Approximation cA5]
    DWT --> Details[Detail Coefficients cD1-cD5]
    
    Details --> Knee[Kneedle Adaptive Thresholding]
    Knee --> Soft[Soft Thresholding]
    
    cA5 --> Recon[Reconstruct Signal]
    Soft --> Recon
    Recon --> Clean[Pristine Denoised ECG]
    
    Clean --> NK2[NeuroKit2: Morphological Features]
    Clean --> WaveStats[Wavelet Statistical Features]
    
    NK2 --> Final[ML-Ready Tabular Data]
    WaveStats --> Final
```

### Key Technical Details:
* **Adaptive Wavelet Denoising**: Employs Discrete Wavelet Transform (DWT) using the `sym8` wavelet. Applies adaptive, knee-point soft thresholding to detail coefficients to suppress high-frequency noise (e.g., muscle artifacts).
* **Baseline Preservation**: Explicitly preserves the low-frequency approximation (`cA5`) to maintain the true baseline wander and foundational signal morphology.
* **Morphological Feature Extraction**: Utilizes `neurokit2` for peak detection and wave delineation to extract critical clinical metrics (e.g., QRS duration, ST-segment deviation, T-wave amplitude, QT interval).

---

## 🧠 3. Dual-Engine AI Consensus

To ensure high diagnostic reliability and avoid false positives, CardioVision combines two different AI paradigms.

```mermaid
flowchart TD
    Data[Processed ECG Data] --> DL[Deep Learning Engine]
    Data --> ML[Machine Learning Engine]
    
    subgraph Engine 1: Deep Learning
        DL --> LSTM[Bidirectional LSTM]
        LSTM -->|Identifies Temporal Sequences| Pred1[Prediction A]
    end
    
    subgraph Engine 2: Machine Learning
        ML --> RF[Random Forest]
        ML --> XGB[XGBoost]
        RF -->|Analyzes Clinical Features| Pred2[Prediction B]
        XGB -.-> Pred2
    end
    
    Pred1 --> Vote{Soft-Voting Consensus}
    Pred2 --> Vote
    
    Vote -- Match --> Confident[High Confidence Diagnosis]
    Vote -- Disagree --> Alert[⚠️ Discrepancy Alert for Clinician]
    
    Confident --> SHAP[SHAP Explainability Visuals]
    Alert --> SHAP
```

### Engine Breakdown:
* **Deep Learning Engine**: A Bidirectional LSTM model (`lstm_model.pth`) analyzes the raw, temporal sequential ECG signal to capture complex time-series dependencies.
* **Machine Learning Engine**: Traditional, highly interpretable classifiers (`rf_model.pkl`, `xgb_model.json`) are trained on tabular morphological features extracted by the pipeline.
* **Discrepancy Alerts**: If the two engines disagree, the soft-voting system highlights the discrepancy for manual clinical review, ensuring safety.

---

## 💾 4. Data Infrastructure & Scale

> [!TIP]
> **Data Scale:** The project evolved from a 1,000-sample prototype to a highly robust **72,000-row balanced dataset** (`ecg_dataset_72k_balanced.csv`) to improve generalization across various patient profiles.

* **Batch Processing**: Scripts like `process_massive_dataset.py` process large-scale data in memory-efficient chunks.
* **Data Augmentation**: Employs Generative Adversarial Networks (`train_gan.py`) to synthesize additional data and balance underrepresented disease classes.
