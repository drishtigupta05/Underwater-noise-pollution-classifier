import librosa
import soundfile as sf
import os
import numpy as np

INPUT_ROOT = "dataset_raw"
OUTPUT_ROOT = "dataset_final"

SR = 16000
TARGET_SEC = 5
TARGET_LEN = SR * TARGET_SEC

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for cls in os.listdir(INPUT_ROOT):
    in_dir = os.path.join(INPUT_ROOT, cls)
    out_dir = os.path.join(OUTPUT_ROOT, cls)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.endswith(".wav"):
            continue

        y, _ = librosa.load(os.path.join(in_dir, fname), sr=SR)

        if len(y) >= TARGET_LEN:
            # split long audio
            for i in range(0, len(y) - TARGET_LEN + 1, TARGET_LEN):
                clip = y[i:i + TARGET_LEN]
                out_name = f"{fname[:-4]}_{i//TARGET_LEN}.wav"
                sf.write(os.path.join(out_dir, out_name), clip, SR)
        else:
            # pad short audio
            pad_width = TARGET_LEN - len(y)
            y_padded = np.pad(y, (0, pad_width))
            sf.write(os.path.join(out_dir, fname), y_padded, SR)
