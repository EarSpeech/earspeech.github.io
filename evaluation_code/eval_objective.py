'''

for eval the model, pesq, stoi, si-sdr

This code mainly refers to a open repository:
    https://github.com/huyanxin/phasen/blob/master/tools/eval_objective.py

Need to install following third-party dependency packages:
    - pypesq: 
        https://github.com/vBaiCai/python-pesq

    - pystoi: 
        https://github.com/mpariente/pystoi.


'''


import torch
import soundfile as sf
from pypesq import pesq
import multiprocessing as mp
import argparse
from pystoi.stoi import stoi
import numpy as np 
import os
from torch_stft import STFT

import librosa.display
import matplotlib.pyplot as plt


os.environ['OMP_NUM_THREADS'] = '2'



def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    assert fs == sr
    if len(wave_data.shape) > 2:
        if wave_data.shape[1] == 1:
            wave_data = wave_data[0]
        else:
            wave_data = np.mean(wave_data, axis=-1)

    return wave_data, fs

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = np.mean(signal)
    signal -= mean
    return signal


def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))


def pow_norm(s1, s2):
    return np.sum(s1 * s2)


def si_sdr(estimated, original):
    estimated = remove_dc(estimated)
    original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))

def eval(ref_name, enh_name, nsy_name, results):
    try:
        
        utt_id = ref_name.split('/')[-1]
        
        enh, d_sr = sf.read(enh_name)
        ref, sr = sf.read(ref_name)
        nsy, sr = sf.read(nsy_name)


        
        ref= librosa.resample(ref, sr, d_sr)
        nsy= librosa.resample(nsy, sr, d_sr)
        
        # enh, sr = audioread(enh_name)
        enh_len = enh.shape[0]
        ref_len = ref.shape[0]
        if enh_len > ref_len:
            enh = enh[:ref_len]
        else:
            ref = ref[:enh_len]
            nsy = nsy[:enh_len]

        ref_score = pesq(ref, nsy, sr)
        enh_score = pesq(ref, enh, sr)
        ref_stoi = stoi(ref, nsy, sr, extended=False)
        enh_stoi = stoi(ref, enh, sr, extended=False)
        ref_sdr = si_sdr(nsy, ref)
        enh_sdr = si_sdr(enh, ref)



    except Exception as e:
        print(e)
    # print(enh_sdr)
    results.append([utt_id, 
                    {'pesq':[ref_score, enh_score],
                     'stoi':[ref_stoi,enh_stoi],
                     'si_sdr':[ref_sdr, enh_sdr]
                    }])

def main(args):
    pathe=args.pathe
    pathc=args.pathc
    pathn=args.pathn
   
    # pool = multiprocessing.Pool(3)
    pool = mp.Pool(args.num_threads)
    mgr = mp.Manager()
    results = mgr.list()
    count=0
    rec_paths = os.listdir(pathe)

    with open(args.result_list, 'w') as wfid:
        for line in rec_paths:
            # print(line)
            count=count+1
            # if count >5:
            #     break
            if ".wav" in line:
                
                name = line
                # print(name)
                pool.apply_async(
                    eval,
                    args=(
                        pathc + name,
                        pathe + name,
                        pathn + name,
                        results,
                    )
                )
        pool.close()
        pool.join()
        print("Start to write to result list!")
        for eval_score in results:
            utt_id, score = eval_score
            pesq = score['pesq']
            stoi = score['stoi']
            si_sdr = score['si_sdr']
            
            wfid.writelines(
                '{:s} {:.3f} {:.3f} '.format(utt_id, pesq[0], pesq[1])
            )
            wfid.writelines(
                '{:.3f} {:.3f} '.format(stoi[0], stoi[1])
            )
            wfid.writelines(
                '{:.3f} {:.3f} '.format(si_sdr[0], si_sdr[1])
            )



    
    

        
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--result_list',
        type=str,
        default='/home/jupyter/EarSpeech_code/SE_model/steps/exp/cldnn/enhancement_performance_on_80000_audio_length.log'
        ) 
    parser.add_argument(
        '--mix_list',
        type=str,
        default='/data/jupyter-data/dataset_speech_dual_channel/data_2/wavs/air_ie_mix_noise_list.lst'
        ) 
    
    parser.add_argument(
        '--num_threads',
        type=int,
        default=20
        )

    parser.add_argument(
        '--pathe',
        type=str,
        default='/home/jupyter/EarSpeech_code/SE_model/steps/exp/cldnn/rec_wav_DC_SE_with_80000_audio_lengths/'
        )
    parser.add_argument(
        '--pathc',
        type=str,
        default='/data/jupyter-data/dataset_speech_dual_channel/data_2/wavs/output_clean_air/'
        )
    parser.add_argument(
        '--pathn',
        type=str,
        default='/data/jupyter-data/dataset_speech_dual_channel/data_2/wavs/output_noisy_air/'
        )
    parser.add_argument(
        '--nanFigureLocation',
        type=str,
        default='./nanFigureLocation/'
        )
    
    args = parser.parse_args()
    avg_snr = 0;
    count = 0;
    with open(args.mix_list,"r") as fid:

        for line in fid:
            # print(line)
            _, _, _, _, snr1 = line.strip().split()
            
            avg_snr = avg_snr + float(snr1)
            count+=1
    fid.close()
    print('---------------')
    print(avg_snr/count)
   
    
    
    if os.path.isfile(args.result_list):
        print("result file is existed")
    else:
        main(args)
    
    if not os.path.isdir(args.nanFigureLocation):
        os.mkdir(args.nanFigureLocation)
    
    en_pesq_with_background_noise=0.0
    en_pesq_with_music_noise=0.0
    en_pesq_with_speech_noise=0.0
    en_pesq_avg=0.0
    
    
    en_stoi_with_background_noise=0.0
    en_stoi_with_music_noise=0.0
    en_stoi_with_speech_noise=0.0
    en_stoi_avg=0.0
    
    en_sdr_with_background_noise=0.0
    en_sdr_with_music_noise=0.0
    en_sdr_with_speech_noise=0.0
    en_sdr_avg=0.0
    
    
    
    raw_pesq_with_background_noise=0.0
    raw_pesq_with_music_noise=0.0
    raw_pesq_with_speech_noise=0.0
    raw_pesq_avg=0.0
    
    
    raw_stoi_with_background_noise=0.0
    raw_stoi_with_music_noise=0.0
    raw_stoi_with_speech_noise=0.0
    raw_stoi_avg=0.0
    
    raw_sdr_with_background_noise=0.0
    raw_sdr_with_music_noise=0.0
    raw_sdr_with_speech_noise=0.0
    raw_sdr_avg=0.0
    
    
    
    background_noise_num = 0
    music_noise_num = 0
    speech_noise_num = 0

    stft = STFT(
                filter_length = 512,
                hop_length = 100,
                win_length = 400,
                window = "hamming",
                    )
    
        
    numlen = 0
    
    with open(args.result_list,"r") as fid:

        for line in fid:
            # print(line)
            flag = 0
            fileName, pesq0, pesq1, stoi0, stoi1, sdr0, sdr1, ssnr0, ssnr1 = line.strip().split()
            
            fileName_segments = fileName.split("_")

            
            
            if np.isnan(float(sdr0)) or np.isnan(float(pesq0)) or np.isnan(float(stoi0)):
                continue;
                
                
            if "music" not in fileName_segments[10] and ".wav" in fileName_segments[10]:
                
                
                en_pesq_with_background_noise += float(pesq1)
                en_stoi_with_background_noise += float(stoi1)
                en_sdr_with_background_noise += float(sdr1)
                
                
                raw_pesq_with_background_noise += float(pesq0)
                raw_stoi_with_background_noise += float(stoi0)
                raw_sdr_with_background_noise += float(sdr0)
                
                background_noise_num+=1              
                
            # Music noise                
            if "music" in fileName_segments[10] and ".wav" in fileName_segments[10]:
                
                en_pesq_with_music_noise += float(pesq1)
                en_stoi_with_music_noise += float(stoi1)
                en_sdr_with_music_noise += float(sdr1)
                
                
                raw_pesq_with_music_noise += float(pesq0)
                raw_stoi_with_music_noise += float(stoi0)
                raw_sdr_with_music_noise += float(sdr0)
                
                music_noise_num+=1    
            
            # Speech noise                
            if ".wav" not in fileName_segments[10]:
                en_pesq_with_speech_noise += float(pesq1)
                en_stoi_with_speech_noise += float(stoi1)
                en_sdr_with_speech_noise += float(sdr1)
                
                raw_pesq_with_speech_noise += float(pesq0)
                raw_stoi_with_speech_noise += float(stoi0)
                raw_sdr_with_speech_noise += float(sdr0)
                speech_noise_num+=1    

            numlen+=1
    fid.close()
    print("haha {:.2f}, {:.2f}, {:.2f}".format(background_noise_num,music_noise_num,speech_noise_num))
    print(numlen)

    print('Raw Average PESQ in Background Noise, Music Noise, Speech noise and average are {:.2f}, {:.2f}, {:.2f}, {:.2f}'
        .format(
        raw_pesq_with_background_noise/background_noise_num, raw_pesq_with_music_noise/music_noise_num, raw_pesq_with_speech_noise/speech_noise_num,
        (raw_pesq_with_background_noise+raw_pesq_with_music_noise+raw_pesq_with_speech_noise)/numlen
    ))
    print('Raw Average STOI in Background Noise, Music Noise, Speech noise and average are {:.2f}, {:.2f}, {:.2f}, {:.2f}'
        .format(
        raw_stoi_with_background_noise/background_noise_num, raw_stoi_with_music_noise/music_noise_num, raw_stoi_with_speech_noise/speech_noise_num,
            (raw_stoi_with_background_noise+raw_stoi_with_music_noise+raw_stoi_with_speech_noise)/numlen
    ))
    print('Raw Average SI-SDR in Background Noise, Music Noise, Speech noise and average are {:.2f}, {:.2f}, {:.2f}, {:.2f}'
        .format(
        raw_sdr_with_background_noise/background_noise_num, raw_sdr_with_music_noise/music_noise_num, raw_sdr_with_speech_noise/speech_noise_num,
            (raw_sdr_with_background_noise+raw_sdr_with_music_noise+raw_sdr_with_speech_noise)/numlen
    ))



    print('Enhanced Average EN PESQ in Background Noise, Music Noise, Speech noise and average are {:.2f}, {:.2f}, {:.2f}, {:.2f}'
        .format(
        en_pesq_with_background_noise/background_noise_num, en_pesq_with_music_noise/music_noise_num, en_pesq_with_speech_noise/speech_noise_num,
        (en_pesq_with_background_noise+en_pesq_with_music_noise+en_pesq_with_speech_noise)/numlen
    ))
    print('Enhanced Average EN STOI in Background Noise, Music Noise, Speech noise and average are {:.2f}, {:.2f}, {:.2f}, {:.2f}'
        .format(
        en_stoi_with_background_noise/background_noise_num, en_stoi_with_music_noise/music_noise_num, en_stoi_with_speech_noise/speech_noise_num,
            (en_stoi_with_background_noise+en_stoi_with_music_noise+en_stoi_with_speech_noise)/numlen
    ))
    print('Enhanced Average EN SI-SDR in Background Noise, Music Noise, Speech noise and average are {:.2f}, {:.2f}, {:.2f}, {:.2f}'
        .format(
        en_sdr_with_background_noise/background_noise_num, en_sdr_with_music_noise/music_noise_num, en_sdr_with_speech_noise/speech_noise_num,
            (en_sdr_with_background_noise+en_sdr_with_music_noise+en_sdr_with_speech_noise)/numlen
    ))

    
    

    


