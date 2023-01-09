import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import librosa
from skimage.util import view_as_windows
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn10, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=32, 
                                               time_stripes_num=2, 
                                               freq_drop_width=8, 
                                               freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
    
    def forward(self, input):
        
        """Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   
        x = self.logmel_extractor(x)    
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        return embedding
    
    
class LaughEventDetection(nn.Module):
    def __init__(self, emb_extractor = None, sample_rate = 32_000, window_size = 512, 
                 hop_size = 50, mel_bins = 64, fmin = 50, fmax = 8000, emb_dim = 512,
                 classes_num = 1):

        super(LaughEventDetection, self).__init__()

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.emb_dim= emb_dim
        self.classes_num = classes_num
        self.emb_extractor = emb_extractor

        self.fc11 = nn.Linear(self.emb_dim, self.classes_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        embedding = self.emb_extractor(inp)
        out = self.fc11(embedding)
        out = self.sigmoid(out)
        return out

warnings.filterwarnings("ignore")
  
SAMPLE_RATE = 32_000
WINDOW_SIZE = 1024
HOP_SIZE = 50
MEL_BINS = 64
FMIN = 8
FMAX = 14_000
AUDIOSET_CLASSES_NUM = 527
CLASSES_NUM = 1
EMB_EXTRACTOR = Cnn10
MIN_DURATION = 0.8
MIN_DIFF_STEP = 20
MIN_DIFF_TIME = 0.2
HOP_LEN = 0.05 
FRAME_LEN = 0.2
THRESHOLD = 0.5
MIN_FRAMES_NUM =  5
device = torch.device("cpu")

def get_model(mymodel, emb_extractor,  *args, **kwargs):
    model = mymodel(emb_extractor, *args, **kwargs).to(device)
    return model

def load_model(checkpoint_path, emb_extractor):
    model = get_model(LaughEventDetection,
                      emb_extractor = emb_extractor,
                      sample_rate=SAMPLE_RATE, 
                      window_size=WINDOW_SIZE,
                      hop_size=HOP_SIZE,
                      mel_bins=MEL_BINS,
                      fmin=FMIN,
                      fmax=FMAX,
                      classes_num=CLASSES_NUM
                     )
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

checkpoint_path = "best_model.pth"
emb_extractor = Cnn10(sample_rate=SAMPLE_RATE, 
                      window_size=WINDOW_SIZE,
                      hop_size=HOP_SIZE,
                      mel_bins=MEL_BINS,
                      fmin=FMIN,
                      fmax=FMAX,
                      classes_num=AUDIOSET_CLASSES_NUM
                      )

model = load_model(checkpoint_path, emb_extractor)
model.eval()

round_tuple = lambda x: (round(x[0], 3), round(x[1], 3))

def time_marks(out_idx, 
               step = MIN_DIFF_STEP, 
               duration = MIN_DURATION, 
               hop_len=HOP_LEN, 
               frame_len=FRAME_LEN):
    
    result_times = []
    start_flag = True
    
    if len(out_idx) > 0: 
        for i in range(1, len(out_idx)):

            if start_flag: # and check_neighbours(out_idx, i):
                start_idx = out_idx[i-1]
                start_flag = False
                
            elif out_idx[i] - out_idx[i-1] <= step:
                continue

            else:
                if not start_flag:
                    end_idx = out_idx[i-1]
                    result_times.append((start_idx*hop_len, end_idx*hop_len + frame_len//2))
                    start_flag = True

        if not start_flag:
            result_times.append((start_idx*hop_len, out_idx[-1]*hop_len + frame_len//2))
            
    result = []
    for item in result_times:
        if item[1] - item[0] >= duration:
            result.append(item)
    
    return list(map(round_tuple, result))


def union_frames(result, min_diff = MIN_DIFF_TIME):
    i, j = 0, 0
    new_result = []
    while i <= len(result)-1 and j <= len(result)-1:
        first = result[i][0]
        second = result[j][1]
        while j < len(result)-1 and result[j+1][0] - second < min_diff:
            second = result[j+1][1]
            j += 1
        new_result.append((first, second))
        j += 1
        i = j
    return new_result


def inference(file_path,  
              step = MIN_DIFF_STEP,
              threshold=THRESHOLD, 
              min_duration = MIN_DURATION,
              min_frames_num = MIN_FRAMES_NUM,
              device = device):
    
    aud, sr = librosa.load(file_path, sr = 32_000)
    duration = len(aud) // sr
    hop_len = int(0.050 * sr)
    win_len = int(0.2 * sr)
    frame_length = int(sr*0.2)
    splited = torch.as_tensor(view_as_windows(aud, window_shape=win_len, step=hop_len)).to(device)
    
    with torch.no_grad():
        output = model(splited).to("cpu").detach().numpy().ravel()
        
    positive_frames = np.where(output>=threshold)[0]
    
    if len(positive_frames) > 5:
        marks = time_marks(positive_frames, duration=min_duration, step = step)      
    else:
        marks = []
    
    
    if len(marks) >= 2:
        union_result = union_frames(marks)
    else:
        union_result = marks
    
    return union_result

def main():
    file_path = sys.argv[1]
    output = inference(file_path)
    print(output)
    return output
    
if __name__ == "__main__":
    main()