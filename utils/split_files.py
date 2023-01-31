import numpy as np
import pandas as pd
import librosa
import torch


CSV_FILE_PATH = "data/targets.csv"
SR = 32_000
FRAMES_DATA_PATH = 'data/frames_data/'
WIN_LEN = int(SR * 0.2)
HOP_LEN = int(SR * 0.05)


def split_files(csv_path=CSV_FILE_PATH, save_folder=FRAMES_DATA_PATH,
                win_len=WIN_LEN, hop_len=HOP_LEN):
    df = pd.read_csv(csv_path)
    files, targets = df.files.values, df.target.values
    for i, file in enumerate(files):
        aud, sr = librosa.load(file, sr=None)

        if aud.shape[0] > 320_000:
            aud = aud[:320_000]

        aud_idx = 0
        target = targets[i]

        for j in range(len(target)-1):
            temp_len = len(aud[aud_idx:aud_idx + win_len])
            file_name = save_folder + file.split("\\")[-1][:-4] + str(np.random.randint(0, 100000)) + "###" + str(target[j])

            if temp_len < win_len:
                pad_len = win_len - temp_len
                temp_aud = np.pad(aud, pad_len, mode="reflect")[pad_len:]
                torch.save(torch.tensor(temp_aud[aud_idx:aud_idx + win_len], dtype=torch.float32),
                           file_name + ".pt")

            else:

                torch.save(torch.tensor(aud[aud_idx:aud_idx + win_len], dtype=torch.float32),
                           file_name + ".pt")

            aud_idx += hop_len


def main():
    split_files()


if __name__ == "__main__":
    main()