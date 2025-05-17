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

## 📄 Description

This project implements a **Speech Emotion Recognition (SER)** system that classifies emotional states from spoken audio using **Mel-spectrograms** and **deep learning architectures**. The core objective is to enhance **human-robot interaction** by enabling machines to understand emotional context from speech.

The system leverages **EfficientNetB7** and **ResNet** models fine-tuned on spectrogram representations of audio signals. It supports both **binary** and **multiclass** emotion classification tasks and includes an end-to-end pipeline from preprocessing to inference.

---

## 🧩 Project Structure and Methodology

The following outlines the core components and their corresponding locations based on the current repository:

1. **Dataset Preprocessing**  
   - Implemented within the notebook:  
     📁 `notebook/Speech_Emotion_Detection.ipynb`  
   - Functions include:
     - Audio loading and trimming  
     - Normalization  
     - Label encoding  

2. **Spectrogram Generation and Model Pipeline**  
   - **Mel-Spectrogram Extraction** and input preparation are carried out in the same notebook.  
   - Feature extraction and augmentation are applied before model training.

3. **Model Training and Testing**  
   - Training is implemented inside the Jupyter notebook.  
   - Models include deep CNNs based on EfficientNet and ResNet variants (training setup visible in `Speech_Emotion_Detection.ipynb`).

4. **Visualization and Evaluation**  
   - Performance metrics and confusion matrix are plotted in the notebook and stored in:  
     📁 `metrics/Final_metrics.png`  
   - Intermediate results and insights are documented in:  
     📁 `metrics/Report_DS.pdf`  

5. **Inference Script**  
   - A standalone Python file for model inference is available:  
     📄 `app.py`  
   - This script loads the trained model and processes new audio inputs for prediction.

---

> **Note:** While this repository does not yet reflect subfolders for binary/multiclass fine-tuning, the entire training and evaluation pipeline is modular and easily extendable to incorporate such separations.



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

