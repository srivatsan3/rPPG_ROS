import sys
import numpy as np
import cv2
import torch
import yaml


with open('src/rppg_node/config/rppg_run_params.yaml','r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

# rppg_tb_path = '/home/mscrobotics2425laptop11/Dissertation/rppgtb/rPPG-Toolbox'
rppg_tb_path = data['rppg_toolbox_node']['ros__parameters']['rppgtb_path']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, rppg_tb_path) # Path to rPPG toolbox

# Importing different non Neural Network methods from the rPPG Toolbox
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.LGI import LGI
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.OMIT import OMIT
from unsupervised_methods.methods.PBV import PBV
from unsupervised_methods.methods.PBV import PBV2

from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.BigSmall import BigSmall

from evaluation.post_process import _calculate_peak_hr, _calculate_fft_hr
from scipy.signal import find_peaks


def estimate_bpm_from_bvp(bvp, fps):
    peaks, _ = find_peaks(bvp, distance=fps * 0.5)  
    duration_sec = len(bvp) / fps
    bpm = (len(peaks) / duration_sec) * 60
    return bpm

def run_rppg(buffer, fps, algo = 'POS', bpm_estimate = 'fft'):
    rppg_algorithms = {
            'chrom': lambda frames: CHROME_DEHAAN(frames, FS = fps),
            'pos': lambda frames: POS_WANG(frames, fs = fps),
            'green': lambda frames: GREEN(frames),
            'ica': lambda frames: ICA_POH(frames, FS = fps),
            'pbv': lambda frames: PBV(frames),
            'pbv2': lambda frames: PBV2(frames),
            'lgi' : lambda frames: LGI(frames),
            'omit' : lambda frames: OMIT(frames)
        }

    bvp = rppg_algorithms[algo](buffer)

    if bpm_estimate == 'fft':
        bpm = _calculate_fft_hr(bvp, fs = fps)
    elif bpm_estimate == 'peak':
        bpm = _calculate_peak_hr(bvp, fs = fps)

    # bpm = estimate_bpm_from_bvp(bvp, fps=fps)

    return bpm

def run_rppg_nn(buffer, fps, model,algo = 'physnet', bpm_estimate = 'fft'):

    inputs, rets = prep_input(buffer, algo)
    with torch.no_grad():
        if rets == -1:
            rppg_signal = model(inputs)
        else:
            rppg_signal = model(inputs)[rets]
    bvp = rppg_signal.detach().cpu().numpy().flatten()

    if bpm_estimate == 'fft':
        bpm = _calculate_fft_hr(bvp, fs=fps)
    elif bpm_estimate == 'peak':
        bpm = _calculate_peak_hr(bvp, fs = fps)
    return bpm


def load_model(algo,frames = 300):
    img_size = 72 # Default value for EfficientPhys and DeepPhys
    if algo == 'efficientphys':
        model = EfficientPhys(frame_depth=frames, img_size=img_size).to(device)
        checkpoint_path = rppg_tb_path+'/final_model_release/UBFC-rPPG_EfficientPhys.pth'
    if algo == 'physnet':
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=frames).to(device)
        checkpoint_path = rppg_tb_path+'/final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth'
    if algo == 'bigsmall':
        model = BigSmall().to(device)
        checkpoint_path = rppg_tb_path+'/final_model_release/BP4D_BigSmall_Multitask_Fold1.pth'
    if algo == 'deepphys':
        model = DeepPhys(img_size = img_size).to(device)
        checkpoint_path = rppg_tb_path+'/final_model_release/UBFC-rPPG_DeepPhys.pth'

    return model, checkpoint_path


def prep_input(buffer, algo):

    if algo == 'efficientphys':
        inputs = prepare_input_for_efficientphys(buffer)
        rets = -1
    elif algo == 'physnet':
        inputs = prepare_input_for_physnet(buffer)
        rets = 0
    elif algo == 'bigsmall':
        inputs = prepare_input_for_bigsmall(buffer)
        rets = 1
    elif algo == 'deepphys':
        inputs = prepare_input_for_deepphys(buffer)
        # print('********',inputs.shape)
        rets = -1

    return inputs, rets




def prepare_input_for_bigsmall(buffer, big_res=144, small_res=9):

    raw_np = np.array(buffer).astype(np.float32) / 255.0  # [T, H, W, C]
    T, H, W, C = raw_np.shape


    big_np = np.stack([cv2.resize(f, (big_res, big_res)) for f in raw_np])  # [T, H_big, W_big, C]

    diff_np = np.diff(big_np, axis=0)  # [T-1, H_big, W_big, C]
    diff_np = np.pad(diff_np, ((1,0),(0,0),(0,0),(0,0)), mode='constant')  # Pad to match length
    diff_np = np.clip(diff_np * 5.0, -1.0, 1.0)  # Rescale motion amplitude

    small_np = np.stack([cv2.resize(f, (small_res, small_res)) for f in diff_np])  # [T, H_small, W_small, C]

    big_tensor   = torch.tensor(big_np).permute(0, 3, 1, 2).to(device)    # [T, C, H_big, W_big]
    small_tensor = torch.tensor(small_np).permute(0, 3, 1, 2).to(device)  # [T, C, H_small, W_small]

    return [big_tensor, small_tensor]

def prepare_input_for_deepphys(buffer):
    T,H,W,C = np.array(buffer).shape
    resized_frames = []
    for t in range(T):
        frame = np.array(buffer)[t]
        resized = cv2.resize(frame, (72,72))
        resized_frames.append(resized)
    resized_frames = np.array(resized_frames)

    raw_np = np.array(resized_frames) / 255.0  # [T, H, W, C]

    diff_np = np.diff(raw_np, axis=0)       # [T-1, H, W, C]
    diff_np = np.pad(diff_np, ((1,0),(0,0),(0,0),(0,0)), mode='constant')  # Pad to match length

    combined_np = np.concatenate([diff_np, raw_np], axis=-1)  # [T, H, W, 6]
    frames_tensor = torch.tensor(combined_np, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # [T, 6, H, W]

    return frames_tensor

def prepare_input_for_efficientphys(buffer):
    frames_np = np.array(buffer) / 255.0  # [T, H, W, C]
    T,H,W,C = frames_np.shape
    resized_frames = []
    for t in range(T):
        frame = frames_np[t]
        resized = cv2.resize(frame, (72,72))
        resized_frames.append(resized)
    resized_frames = np.array(resized_frames)

    frames_tensor = torch.tensor(resized_frames, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    B, C, T, H, W = frames_tensor.shape
    inputs = frames_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [300, 3, 96, 96]

    return inputs

def prepare_input_for_physnet(buffer):
    frames_np = np.array(buffer) / 255.0  # [T, H, W, C]
    frames_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    return frames_tensor
   