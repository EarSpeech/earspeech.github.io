


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
from matplotlib import pyplot as plt
import soundfile as sf
import librosa

def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)**0.5
    
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)

        outputs = F.conv1d(inputs, self.weight, stride=self.stride)
         
        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim//2+1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2+imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.win_inc = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 

        if phase is not None:
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
            # print(inputs.shape)
        # print(inputs.shape)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride) 

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1,1,inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs/(coff+1e-8) 
        return outputs

def test_fft(win_len, win_inc, fft_len, fileName):
    torch.manual_seed(20)
    # win_len = 320
    # win_inc = 80
    # fft_len = 1024
    # inputs = torch.randn([1, 1, 16000*5])
    # print(inputs.shape)
    inputs = sf.read(fileName)[0]
    print(inputs.shape)
    inputs = torch.tensor(inputs)
    print(inputs.shape)

    inputs = inputs.to(torch.float32)
    inputs = torch.unsqueeze(inputs,0)
    inputs = torch.unsqueeze(inputs, 0)
    print(inputs.shape)

    stft_test = ConvSTFT(win_len, win_inc, fft_len, win_type='hanning', feature_type='real')

    outputs1 = stft_test(inputs)[0]
    print(outputs1.shape)
    outputs1 = outputs1.numpy()[0]
    np_inputs = inputs.numpy().reshape([-1])
    # librosa_stft = librosa.stft(np_inputs, win_length=win_len, n_fft=fft_len, hop_length=win_inc, center=False)
    # papap = np.abs(librosa_stft)
    # print(papap.shape)


    # fig, (ax0, ax1) = plt.subplots(2, 1)
    # c = ax0.pcolor(papap, cmap='jet')
    # ax0.set_title('default: no edges')
    # c = ax1.pcolor(outputs1, cmap='jet')
    # ax1.set_title('thick edges')

    # fig.colorbar(ax0)
    # fig, ax = plt.subplots(figsize=(12, 6))
    # c = ax.pcolor(papap, cmap='jet')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency')
    # plt.show()
    # print(np.mean((outputs1 - np.abs(librosa_stft))**2))


def test_ifft1():

    N = 400
    inc = 100
    fft_len = 512
    torch.manual_seed(N)
#    inputs = torch.randn([1, 1, N*3000])
    inputs = sf.read('../ljm_arctic_a0001.wav')[0]
    inputs = inputs.reshape([1,1,-1])
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='real')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='real')
    
    inputs = torch.from_numpy(inputs.astype(np.float32))
    print(inputs.shape)
    outputs1,phase_output = fft(inputs)
    outputs2 = ifft(outputs1, phase_output)
    print(outputs2.shape)

    sf.write('conv_stft.wav', outputs2.numpy()[0,0,:],16000)
    print('wav MSE', torch.mean(torch.abs(inputs[...,:outputs2.size(2)]-outputs2)))
    fig, (ax0, ax1) = plt.subplots(1, 2)


    ax0.plot(outputs2.numpy()[0,0,:])

    ax1.plot(inputs.numpy()[0,0,:])

    # fig.colorbar(ax0)
    fig.tight_layout()
    plt.show()

def test_ifft2():
    N = 400
    inc = 100
    fft_len=512
    np.random.seed(20)
    torch.manual_seed(20)
    t = np.random.randn(16000*4)*0.005
    t = np.clip(t, -1, 1)
    #input = torch.randn([1,16000*4]) 
    input = torch.from_numpy(t[None,None,:].astype(np.float32))
    
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    
    out1 = fft(input)

    output = ifft(out1)
    print('random MSE', torch.mean(torch.abs(input-output)**2))
    import soundfile as sf
    sf.write('zero.wav', output[0,0].numpy(),16000)


if __name__ == '__main__':
    test_fft(320,80,1024, '../dual_channel_analysis_air_audio.wav')
    # test_fft(320, 80, 1024, '../dual_channel_analysis_inear_audio.wav')
    # inputs = sf.read('../ljm_arctic_a0001.wav')[0]
    # test_ifft1()
    #test_ifft2()
