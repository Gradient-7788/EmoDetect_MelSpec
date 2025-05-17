# Speech Emotion Recognition model using audio mel-spectrograms

This work is done by Team Gradient as part of course project for DA221M Aug-Nov 2024 IIT Guwahati.
This project focuses on detecting emotions from speech using deep learning and audio preprocessing techniques like Mel-Spectrogram extraction. It combines exploratory analysis, model training, and result visualization for robust emotion classification.

---

## 📂 Directory Structure

```bash
EmoDetect_MelSpec/
│
├── metrics/                        # Visual results and final evaluation
│   ├── Final_metrics.png
│   └── Report_DS.pdf
│
├── notebook/                       # Jupyter notebook for training/experimentation
│   └── Speech_Emotion_Detection.ipynb
│
├── Dataset Links                   # Text file or markdown linking data sources
├── README.md                       # Project overview and documentation
├── Testing_model.jpg              # Model architecture or testing visualization
├── app.py                          # Script for running predictions/inference
└── requirements.txt
   ```

---

## Description 
This project focuses on developing a speech emotion recognition system using spectrograms and deep learning models, specifically EfficientNetB7 and ResNet, to classify emotions from audio data. The goal is to enhance human-robot interaction by accurately recognizing and understanding emotional states expressed through speech.

The division of methodology, and the corresponding location of code:
1. Dataset Preprocessing: ./PreprocessingInput/
2. Generation of spectrograms and integration of pipeline:
    1. For Multiclass classification task: ./FineTuneCNNs/Multiclass/ contains code of fine tuning of the various different models
    2. For Binary classification task: ./FineTuneCNNs/Binary/ contains code of fine tuning of the various different models
3. Audio Augmentation Techniques employed:
    1. For Data augmentation task: ... contains the code of various augmentation techniques employed
5. Training, Confusion Matrix and output generation:
    1. Generate predictions on all testing dataset created out of the overall dataframe.
    2. Encodings and Decodings (one hot) task: ... contains the overall total training, encodings and confusion matrix codes.

---

## 📊 Features

- 🎧 **Mel-Spectrogram Feature Extraction** for rich audio representation  
- 🔍 **Preprocessing pipeline** including normalization and trimming  
- 🧠 **Deep learning-based model** trained on emotional speech samples  
- 📈 **Visualization** of performance metrics and confusion matrix  
- 🧪 **Interactive notebook** for training, testing, and analysis
-  DFTs, FFTs,
-  Mel-Spectrograms,
-  Singular Value Decomposition
-  EfficientNetB7

---

## 🛠️ Tech Stack

- Python 3.x  
- Librosa  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  
- TensorFlow / Keras  
- Jupyter Notebook  

---

## 📌 Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/Gradient-7788/EmoDetect_MelSpec.git
cd EmoDetect_MelSpec
  ```

### 2. Install Dependencies:

```bash
pip install -r requirements.txt
  ```

### 3. Run the notebook:

- Open notebook/Speech_Emotion_Detection.ipynb in Jupyter Notebook or JupyterLab
- Follow the notebook cells to preprocess audio, train, and evaluate the model

### 4. Run Inference:

```bash
python app.py
  ```

## 📜 License

MIT License. See LICENSE for details.

👥 Contributor

Yesh Lohchab
yesh.3119@iitg.ac.in

