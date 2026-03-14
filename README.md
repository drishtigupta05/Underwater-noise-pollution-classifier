# Underwater-noise-pollution-classifier

This project focuses on **automatic classification of underwater acoustic signals** using machine learning. The goal is to identify different types of sounds recorded by hydrophones and categorize them into meaningful classes such as marine animal vocalizations, human-generated ocean noise, and sonar signals.

The project demonstrates a complete machine learning workflow including **data collection, preprocessing, feature extraction, model training, evaluation, and dataset publication**.

---

## Dataset

The dataset consists of curated hydrophone audio recordings collected from the NOAA Sanctuary Soundscapes Monitoring Project passive acoustic archive.

Audio segments were manually selected from raw recordings and converted into **5-second WAV clips**. Each clip is labeled into one of the following categories:

* **Animal** – Marine animal vocalizations such as whale or dolphin sounds
* **Anthropogenic** – Human-generated ocean noise (ships, machinery, etc.)
* **Sonar** – Active sonar signals

The dataset is publicly available on Kaggle @ https://www.kaggle.com/datasets/drishtigupta73/underwater-noise/data

---

## Project Pipeline

The workflow used in this project includes:

1. **Data Collection**

   * Hydrophone recordings sourced from NOAA SanctSound archive

2. **Preprocessing**

   * Audio segmentation into 5-second clips
   * Resampling and normalization

3. **Feature Extraction**

   * Mel spectrograms / MFCC features

4. **Model Training**

   * Classification models trained to distinguish sound categories

5. **Evaluation**

   * Accuracy
   * Confusion matrix
   * Precision / Recall / F1 score

---

## Technologies Used

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## Applications

Automatic underwater sound classification has several important applications:

* Marine wildlife monitoring
* Detection of human noise pollution in oceans
* Passive acoustic monitoring systems
* Environmental and ecological research

---

## Future Improvements

* Increase dataset size for improved model performance
* Experiment with advanced deep learning models
* Deploy lightweight models for real-time underwater monitoring

---

## License

This project uses curated subsets of publicly available hydrophone recordings from the NOAA Sanctuary Soundscapes Monitoring Project and is intended for **research and educational purposes**.

---

## Contact

Name: Drishti Gupta

Email: drishtigupta700@gmail.com

---

A small scientific thought to end on: hydrophones reveal that the ocean is not a silent world but a **dense acoustic ecosystem**—a mixture of biological signals and human activity. Teaching machines to interpret those signals is a step toward understanding and protecting that hidden soundscape.
