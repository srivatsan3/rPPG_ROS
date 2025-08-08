import sys
rppg_tb_path = '/home/mscrobotics2425laptop11/Dissertation/rppgtb/rPPG-Toolbox'
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
            'ica': lambda frames: ICA_POH(frames),
            'pbv': lambda frames: PBV(frames),
            'pbv2': lambda frames: PBV2(frames),
            'lgi' : lambda frames: LGI(frames),
            'omit' : lambda frames: OMIT(frames)
        }

    bvp = rppg_algorithms[algo](buffer)

    if bpm_estimate == 'fft':
        bpm = _calculate_fft_hr(bvp, fs = fps)

    # bpm = estimate_bpm_from_bvp(bvp, fps=fps)

    return bpm
        # bpm = _calculate_fft_hr(bvp, fs = self.fps)
        # bpm = _calculate_peak_hr(bvp, fs = self.fps)