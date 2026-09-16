# 🌾 Rice Grain Classifier

A Convolutional Neural Network (CNN) that identifies the variety of a rice grain from a photo, wrapped in an interactive Streamlit web app.

**🔗 Live App:** [rice-classification-app.streamlit.app](https://rice-classification-app-cefjy42wzt2bba9jfd4zsh.streamlit)

---

## 📋 Overview

This project trains a CNN from scratch on the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) from Kaggle, then serves the trained model through a Streamlit interface where users can upload a photo of a rice grain or paste an image URL to get an instant variety prediction.

**How it works:**
1. The CNN scans the image in small patches (filters) to detect simple patterns first — edges, shading, and grain outline.
2. Each layer stacks on the previous one, combining simple patterns into more complex shape features (grain length, curvature, translucency) the deeper it goes.
3. At the end, it takes all the learned features and votes on which of the 5 rice varieties the grain most likely belongs to.

## 🏷️ Classes

The model recognizes 5 rice varieties:

`Arborio` · `Basmati` · `Ipsala` · `Jasmine` · `Karacadag`

## 🖥️ App Preview

- Upload an image file (JPG/PNG) **or** paste an image URL
- View the top predicted variety with a confidence score
- See a horizontal bar chart of confidence across all 5 varieties
- Low-confidence predictions (< 60%) are flagged as "not confidently recognized" instead of a false-confident guess

## 🧠 Model Architecture

Built with `tf.keras.Sequential`, trained on 150×150 RGB images:

| Block | Layers |
|-------|--------|
| Input | `Conv2D(32, 3x3, relu)` → `MaxPooling2D(2,2)` |
| Block 2 | `Conv2D(64, 3x3, relu)` → `MaxPooling2D(2,2)` |
| Block 3 | `Conv2D(128, 3x3, relu)` → `MaxPooling2D(2,2)` |
| Block 4 | `Conv2D(264, 3x3, relu)` → `MaxPooling2D(2,2)` |
| Head | `Flatten` → `Dense(512, relu)` → `Dropout(0.5)` → `Dense(5, softmax)` |

**Training configuration:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10
- Batch size: 32
- Augmentation: rotation (±20°), zoom (0.2), horizontal flip, rescaling (1/255) — applied via `ImageDataGenerator`
- Data split: 80% train / 10% validation / 10% test (via `split-folders`)
- Checkpointing: best model saved automatically based on `val_accuracy`

**Results (final epoch):**
- Training Accuracy: **~98.9%**
- Validation Accuracy: **~93.0%**
- Validation Loss: **~0.41**

> ℹ️ Validation accuracy fluctuated across epochs (ranging roughly 84%–98%) before settling around 93% by epoch 10, which is typical for a model without early stopping or a learning-rate schedule. The gap between training (~99%) and validation (~93%) accuracy suggests mild overfitting — see [Future Improvements](#-future-improvements) for ways to tighten this up.

## ⚠️ Important Preprocessing Note

Unlike models built with a built-in `Rescaling` layer, this model was trained using `ImageDataGenerator(rescale=1./255)`, which normalizes pixel values **outside** the model graph. This means any code that loads `rice_cnn_model.keras` for inference — including `app.py` — **must manually divide pixel values by 255** before prediction, or the model's outputs will be meaningless. This is already handled in `app.py`'s `preprocess()` function.

## 📁 Project Structure

```
.
├── test_images/
├── CNN_Rice_Image_Dataset.ipynb   # Data prep, model training & evaluation
├── app.py                         # Streamlit web app
├── rice_cnn_model.keras           # Trained model weights
├── requirements.txt               # Python dependencies
└── README.md
```

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/dawoodofficial04/rice-classification-app.git>
   cd rice-classification-app
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

## 🏋️ Retraining the Model

The full training pipeline is in `CNN_Rice_Image_Dataset.ipynb`:

1. Downloads the dataset from Kaggle via `kagglehub`
2. Splits it 80/10/10 into train/val/test using `split-folders`
3. Loads images with `ImageDataGenerator`, applying rescaling and augmentation to the training set
4. Builds and trains the CNN described above, checkpointing the best model by validation accuracy
5. Saves the final model as `rice_cnn_model.keras`

To retrain, open the notebook, run all cells, and replace `rice_cnn_model.keras` with the newly saved model. If you change how the model normalizes pixels (e.g. by adding a `Rescaling` layer), remember to update `preprocess()` in `app.py` to match.

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **App:** Streamlit
- **Image handling:** Pillow, `tf.keras.preprocessing.image`
- **Data prep:** kagglehub, split-folders, `ImageDataGenerator`
- **Deployment:** Streamlit Community Cloud

## 🚧 Future Improvements

- Add early stopping and a learning-rate scheduler to smooth out the validation accuracy swings seen during training
- Evaluate on the held-out test set explicitly and report test accuracy/loss (not just validation)
- Add a confusion matrix to see which rice varieties are most often confused with each other
- Reduce the Dense(512) layer size or add more Dropout/regularization to narrow the train/validation accuracy gap
- Migrate preprocessing into the model itself (a `Rescaling` layer) to remove the manual `/255` step and reduce the risk of preprocessing mismatches

## 📄 License

Add your license of choice here (e.g. MIT).