# Music Genre Classification App

A research-based full-stack web application that classifies music genres using both **machine learning** and **deep learning** models. The project compares traditional algorithms like SVM and k-NN with CNN and CNN+BiLSTM architectures on two benchmark datasets: **GTZAN** and **FMA-Small**.

---

## Highlights

- **Dual-model evaluation:** ML vs DL for music genre classification  
- **Deep Learning Models:** CNN and hybrid CNN + BiLSTM with SE block and Attention  
- **High Accuracy:** Up to 93% classification accuracy on FMA-Small  
- **Feature Extraction:** MFCCs and Mel Spectrograms  
- **User Interface:** Django-based frontend with registration, login, and history tracking  
- **REST API:** For real-time genre predictions

---

## Datasets

- **GTZAN Dataset**: 1000 tracks across 10 genres  
- **FMA-Small**: 8000 30-second high-quality audio clips  
- All data is preprocessed into `.wav` format and converted to **128x128 Mel spectrograms**.

---

## Technologies Used

### Backend
- Scikit-learn
- TensorFlow / Keras

### Frontend
- HTML/CSS/Bootstrap (with Django templates)
- JavaScript

### ML & DL Models
- Logistic Regression, k-NN, SVM
- CNN
- CNN + BiLSTM + Attention + SE Block

---

## Parameters

### Input Representation
- **Audio Length**: 30 seconds  
- **Sampling Rate**: 44,100 Hz  
- **Feature Type**: Mel-spectrogram  
- **Mel-Spectrogram Shape**: `128 (mel bands) × 128 (time frames)`  
- **Color Channel**: Grayscale (`1 channel`)  
- **Final Input Shape**: `(128, 128, 1)`

---

### CNN Model (FMA-Small)
- **Total Layers**: 5 (Convolution + Pooling + Dense)
- **Dropout Rate**: 45%
- **Activation Functions**: ReLU (hidden), Softmax (output)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Output Classes**: 8 genres
- **Best Accuracy**: 92%

---

### CNN + BiLSTM Model (GTZAN & FMA-Small)
- **CNN Layers**: 4 blocks with increasing filters (32 → 256)
- **LSTM**: Bidirectional LSTM with 128 units
- **Attention**: Applied after BiLSTM
- **Dropout Rate**: 50%
- **Output Classes**:
  - GTZAN: 10 genres
  - FMA-Small: 8 genres
- **Best Accuracy**:
  - GTZAN: 91%
  - FMA-Small: 93%

---

### Classical Machine Learning Models
- **Input Features**: MFCC vectors or flattened spectrograms
- **Best Hyperparameters**:
  - **SVM**: RBF kernel, gamma = scale
  - **k-NN**: 5 neighbors, Manhattan distance, distance-based weight
  - **Logistic Regression**: `C=10`, L2 penalty, saga solver
- **Output Classes**: 8 or 10 (based on dataset)
- **Accuracy Range**: 57%–76%

## Experimental Results & Performance

The performance of the models was evaluated through a rational analysis using **Precision**, **Recall**, and **F1-Score** metrics.

### Table II: GTZAN Dataset Results
The comparative analysis on the GTZAN dataset demonstrates the superior stabilization and performance of deep learning architectures.

| Model Category | Model | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Traditional ML** | k-NN | 0.74 | 0.74 | 0.74 |
| | SVM | 0.74 | 0.75 | 0.74 |
| | Logistic Regression | 0.75 | 0.76 | 0.75 |
| **Deep Learning** | CNN | 0.91 | 0.91 | 0.91 |
| | **CNN + BiLSTM** | **0.91** | **0.91** | **0.91** |

![GTZAN Confusion Matrix](Music%20Genre%20Classification/assets/gtzan_cnn_and_bilstm.jpeg)

### Table III: FMA-Small Dataset Results
On the FMA-Small dataset, the CNN + BiLSTM hybrid architecture proved to be the most effective model for capturing complex data patterns.

| Model Category | Model | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Traditional ML** | k-NN | 0.51 | 0.49 | 0.48 |
| | SVM | 0.62 | 0.61 | 0.61 |
| | Logistic Regression | 0.57 | 0.57 | 0.57 |
| **Deep Learning** | CNN | 0.92 | 0.92 | 0.92 |
| | **CNN + BiLSTM** | **0.93** | **0.93** | **0.93** |

![FMA-Small Confusion Matrix](Music%20Genre%20Classification/assets/fma_small_cnn_and_bilstm.jpeg)

## Training Settings

### Dataset Processing
- **Datasets Used**:
  - GTZAN (10 genres × 100 tracks)
  - FMA-Small (8 genres × 8000 samples)
- **Preprocessing Steps**:
  - Convert audio to `.wav`
  - Normalize audio levels
  - Extract 128×128 Mel-spectrograms
  - Store as `.npy` or tensor images
- **Data Augmentation** (for deep models):
  - Time shifting
  - Pitch shifting
  - Additive Gaussian noise

---

### Training Configuration

| Setting                 | Value                         |
|-------------------------|-------------------------------|
| Epochs (CNN)            | 50                            |
| Epochs (CNN + BiLSTM)   | 50                            |
| Batch Size              | 32                            |
| Optimizer               | Adam                          |
| Learning Rate           | 0.001                         |
| Loss Function           | Categorical Crossentropy      |
| Evaluation Metrics      | Accuracy, Precision, Recall, F1 Score |
| Validation Split        | 20% (Stratified)              |
| Learning Rate Scheduler | `ReduceLROnPlateau` (patience=3, factor=0.5) |
| Model Saving            | `ModelCheckpoint` (save_best_only=True) |
| Early Stopping          | `EarlyStopping` (patience=7, restore_best_weights=True) |

---

### Model Saving & Evaluation
- **Saved Format**: `.h5` (Keras HDF5 format)
- **Model Selection**: Automatically saved best model during training
- **Callback Tools Used**:
  - `EarlyStopping` for preventing overfitting
  - `ModelCheckpoint` for best-model saving
  - `ReduceLROnPlateau` for adaptive learning rate
- **Final Evaluation**: Performed on held-out test set
- **Visualization**: Confusion matrix plotted for all experiments


