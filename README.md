# 🎙️ Direction Voice Command Classifier

A lightweight deep learning model that classifies spoken directional commands — **left**, **right**, **forward**, and **backward** — using MFCC audio features and a compact CNN architecture. Achieves **~96% accuracy** and exports to a **~19.5 KB TFLite model** suitable for edge devices.

---

## 📌 Project Description

This project builds a real-time-ready voice command recognition system trained on the [Google Speech Commands Dataset v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands). It is designed for robotics, embedded navigation, or any application where spoken direction commands need to be recognized efficiently.

The pipeline covers the full ML workflow:

- **Audio loading** at 16 kHz with `librosa`
- **Feature extraction** via 13-coefficient MFCC
- **Normalization & padding** for fixed-size inputs
- **CNN training** with TensorFlow/Keras
- **Evaluation** with classification report and F1-score
- **TFLite export** for on-device inference (~19.5 KB)

---

## 📊 Dataset

| Class      | Samples |
|------------|---------|
| `left`     | 3,801   |
| `right`    | 3,778   |
| `forward`  | 1,557   |
| `backward` | 1,664   |
| **Total**  | **10,800** |

Download via:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
```

---

## ⚙️ Installation

```bash
pip install tensorflow tensorflow-io librosa scikit-learn matplotlib numpy
```

---

## 🔁 Pipeline Overview

### 1. Audio Loading

```python
import librosa

audio, sr = librosa.load("left/sample.wav", sr=16000)
print("Shape:", audio.shape)   # (16000,)
print("Sample rate:", sr)      # 16000
```

---

### 2. MFCC Feature Extraction

```python
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print("MFCC Shape:", mfcc.shape)  # (13, ~32)
```

---

### 3. Padding & Normalization

```python
import numpy as np

MAX_LEN = 32

def pad_mfcc(mfcc, max_len=MAX_LEN):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = pad_mfcc(mfcc)
    return mfcc  # shape: (13, 32)
```

Normalization (applied per sample before saving):

```python
normalized_mfcc = (mfcc_features - np.mean(mfcc_features)) / np.std(mfcc_features)
np.save(output_path, normalized_mfcc)
```

---

### 4. Dataset Preparation

```python
import os
import numpy as np
import tensorflow as tf

classes = ["left", "right", "forward", "backward"]
data_path = "mfcc_data"

X, y = [], []

for label, cls in enumerate(classes):
    class_path = os.path.join(data_path, cls)
    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            mfcc = np.load(os.path.join(class_path, file))
            X.append(mfcc)
            y.append(label)

X = np.array(X)[..., tf.newaxis]   # shape: (N, 13, 32, 1)
y = np.array(y)[..., tf.newaxis]   # shape: (N, 1)
```

Train/test split:

```python
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
# (8640, 13, 32, 1) (2160, 13, 32, 1)
```

---

### 5. Model Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(13, 32, 1)),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Total trainable parameters: 10,404 (~40 KB)**

---

### 6. Training

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=16,
    epochs=20,
)
```

---

### 7. Evaluation

```python
from sklearn.metrics import classification_report, f1_score

y_pred = model.predict(X_test).argmax(axis=1)
f1 = f1_score(y_test, y_pred, average='weighted')

print("F1-score (weighted):", f1)
print(classification_report(y_test, y_pred, target_names=classes))
```

**Results:**

```
              precision    recall  f1-score   support

        left       0.96      0.97      0.97       745
       right       0.98      0.95      0.96       729
     forward       0.93      0.96      0.94       353
    backward       0.94      0.95      0.95       333

    accuracy                           0.96      2160
   macro avg       0.95      0.96      0.95      2160
weighted avg       0.96      0.96      0.96      2160

F1-score (weighted): 0.9584
```

---

### 8. Saving the Model

```python
model.save('speech_command_model.h5')
```

---

### 9. Inference (Keras)

```python
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc[..., np.newaxis]        # (13, 32, 1)
    mfcc = np.expand_dims(mfcc, axis=0) # (1, 13, 32, 1)
    return mfcc

def predict_audio(file_path, model):
    mfcc = preprocess_audio(file_path)
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return classes[predicted_class], confidence

label, conf = predict_audio("backward.ogg", model)
print("Prediction:", label)      # backward
print("Confidence:", conf)       # 0.9935
```

---

### 10. TFLite Export & Inference

**Export:**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite size:", os.path.getsize("model.tflite") / 1024, "KB")
# TFLite size: 19.48 KB
```

**Inference:**

```python
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(file_path):
    mfcc = preprocess_audio(file_path).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    confidence = np.max(output)
    return classes[predicted_class], confidence

label, conf = tflite_predict("left.ogg")
print("Prediction:", label)    # left
print("Confidence:", conf)     # 0.9843
```

---

## 📁 Project Structure

```
direction-voice-commands/
│
├── direction_voice_commands_model.ipynb   # Main notebook
├── mfcc_data/                             # Preprocessed MFCC features
│   ├── left/
│   ├── right/
│   ├── forward/
│   └── backward/
├── speech_command_model.h5                # Saved Keras model
└── model.tflite                           # Optimized TFLite model (~19.5 KB)
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| `TensorFlow 2.x` | Model training & TFLite export |
| `Librosa` | Audio loading & MFCC extraction |
| `NumPy` | Array operations |
| `scikit-learn` | Train/test split, evaluation metrics |
| `Matplotlib` | Waveform & MFCC visualization |

---

## 📈 Model Performance Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | **96%** |
| Weighted F1-Score | **0.9584** |
| Total Parameters | **10,404** |
| Keras Model Size | ~40 KB |
| TFLite Model Size | **~19.5 KB** |

---

## 🚀 Use Cases

- Robotic direction control via voice
- Embedded voice navigation systems
- Smart home/IoT gesture-free control
- Edge inference on microcontrollers or mobile devices

---

## 📄 License

This project uses data from the [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209), released under Creative Commons Attribution 4.0.
