import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "dataset_final"
SR = 16000
N_MFCC = 20

CLASS_MAP = {
    "animal": 0,
    "anthropogenic": 1,
    "sonar": 2
}

def extract_features(path):
    y, _ = librosa.load(path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)

    # Mean + std pooling
    feat = np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ])

    return feat

X, y = [], []

for cls, label in CLASS_MAP.items():
    folder = os.path.join(DATASET_PATH, cls)
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            X.append(extract_features(os.path.join(folder, f)))
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

class_weights = {0:1.0,1:1.0,2:2.0}

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight=class_weights
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print(classification_report(
    y_val,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Animal", "Anthropogenic", "Sonar"],
    zero_division=0
))

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Animal", "Anthropogenic", "Sonar"],
    yticklabels=["Animal", "Anthropogenic", "Sonar"]
)
plt.title("Confusion Matrix – MFCC + SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
