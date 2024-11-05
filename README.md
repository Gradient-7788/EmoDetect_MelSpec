# Speech Emotion Recognition model using audio mel-spectrograms
This work is done by Team Gradient as part of course project for DA221M Aug-Nov 2024 IIT Guwahati.

Team Members: <br>
    Yesh Lohchab 230102115 \
    Vanshika Mittal 230102109

This directory contains the code for the project titled: 

    Speech Emotion Recognition model using audio mel-spectrograms

All these notebooks were run on Jupyter Notebook with the appropriate settings, all links to these publicly avalaible notebooks are located in the cooresponding folders.

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

### Installation

1. Clone or download the repository:
   ```sh
   git clone [https://github.com/surankan-de/DA221M](https://github.com/Gradient-7788/Gradient/blob/main/app.py)
   ```
2. Navigate to the project directory
3. To run the streamlit application:
  ```sh
      streamlit run app.py
  ```
## Features 
1. Predicts the mood of the speaker using the speech.
## Machine Learning/ AI Concepts Used 
1. DFTs, FFTs,
2. Mel-Spectrograms,
3. Singular Value Decomposition
4. EfficientNetB7
