import os
import glob
import numpy as np
import pandas as pd
import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
import torch


LAUGHTER_CLASSES_IDX = range(16, 22)
ROW_DATA_ROOT = "data/raw/"
CSV_FILE_PATH = "data/targets.csv"
SR = 32_000
COEF = 20
HOP_LEN = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sed = SoundEventDetection(checkpoint_path=None, device=device)


def change_class_idx(lst: list):
    """
    Change multiclass classification result in binary one: assign 1 for laugh classes 
    and 0 for all others.
    """
    for idx, item in enumerate(lst):
        if item in LAUGHTER_CLASSES_IDX:
            lst[idx] = 1
        else:
            lst[idx] = 0


def is_laugh_on_frame(frame: list):
    """determine the precende of laugh on the wide frame"""
    return 1 if np.sum(frame) > 0 else 0


def get_targets(row_data_root=ROW_DATA_ROOT):
    """
    Get targets for frames for audios in the root folder.
    """
    
    all_files = []
    targets = []
    folder_names = os.listdir(ROW_DATA_ROOT)
    
    for folder_name in folder_names:

        audio_files = glob.glob(row_data_root + folder_name + "/*")
        all_files.extend(audio_files)
        for file in audio_files:
            (audio, _) = librosa.core.load(file, sr=SR)
            audio = audio[None, :]  # (batch_size, segment_samples)

            framewise_output = sed.inference(audio)
            result = np.argmax(framewise_output[0], axis = 1)
            change_class_idx(result)
            result = [is_laugh_on_frame(result[i:i+COEF]) for i in range(0, len(result)-1, HOP_LEN)]
            targets.append(result)

    df = pd.DataFrame({"files": all_files, "target": targets})
    df.to_csv(CSV_FILE_PATH)


def main():
    get_targets(row_data_root=ROW_DATA_ROOT)


if __name__ == "__main__":
    main()
