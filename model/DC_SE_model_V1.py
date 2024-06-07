


import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(__file__))
from conv_stft import ConvSTFT, ConviSTFT



class FTB(nn.Module):

    def __init__(self, input_dim=257, in_channel=9, r_channel=8):
        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(r_channel),
            nn.ReLU()
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]: Dim = frequency bins; Time = time bins
        '''
        # T-F attention

        conv1_out = self.conv1(inputs)
        B, C, D, T = conv1_out.size()
        reshape1_out = torch.reshape(conv1_out, [B, C * D, T])

        conv1d_out = self.conv1d(reshape1_out)

        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)

        # tranpose to [B,C,D,T]
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs



class ACFeatureEmbedding(nn.Module):

    def __init__(self, input_dim=257, channel_amp=8):
        super(ACFeatureEmbedding, self).__init__()

        self.ac_conv0 = nn.Sequential(
            nn.Conv2d(1, channel_amp,
                      kernel_size=[5, 1],
                      padding=(2, 0)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp,
                      kernel_size=[1, 5],
                      padding=(0, 2)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp,
                      kernel_size=[5, 5],
                      padding=(2, 2)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )
        self.ftb1 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )

        self.ftb2 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )

        self.ftb3 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(channel_amp, 16, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )


    def forward(self, Sac):
        '''
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]

        '''
        ac_out0 = self.ac_conv0(Sac)

        ac_in1 = ac_out0
        ac_out1 = self.ftb1(ac_in1)

        ac_in2 = ac_in1 + ac_out1
        ac_out2 = self.ftb2(ac_in2)

        ac_in3 = ac_out2 + ac_in2
        ac_out3 = self.ftb3(ac_in3)

        ac_en_out1 = self.encoder_conv1(ac_out3)
        ac_en_out2 = self.encoder_conv2(ac_en_out1)
        ac_en_out3 = self.encoder_conv3(ac_en_out2)
        ac_en_out4 = self.encoder_conv4(ac_en_out3)


        return ac_en_out1, ac_en_out2, ac_en_out3, ac_en_out4


class IEFeatureEmbedding(nn.Module):

    def __init__(self, input_dim=257, channel_amp=8):
        super(IEFeatureEmbedding, self).__init__()

        self.ie_conv0 = nn.Sequential(
            nn.Conv2d(1, channel_amp,
                      kernel_size=[5, 1],
                      padding=(2, 0)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp,
                      kernel_size=[1, 5],
                      padding=(0, 2)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp,
                      kernel_size=[5, 5],
                      padding=(2, 2)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )

        self.ftb1 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )

        self.ftb2 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )

        self.ftb3 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )

        self.ie_conv1 = nn.Sequential(
            nn.Conv2d(channel_amp, 16, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )



    def forward(self, Sie):
        '''
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]

        '''

        ie_out0 = self.ie_conv0(Sie)

        ie_in1 = ie_out0
        ie_out1 = self.ftb1(ie_in1)

        ie_in2 = ie_in1 + ie_out1
        ie_out2 = self.ftb2(ie_in2)

        ie_in3 = ie_out2 + ie_in2
        ie_out3 = self.ftb3(ie_in3)

        ie_out4 = self.ie_conv1(ie_out3)

        return ie_out4

class SENN(nn.Module):
    def __init__(self, channel_amp=9):
        super(SENN, self).__init__()
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=[1, 1], padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.skip_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=[1, 1], padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.skip_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=[1, 1], padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.decoder_conv4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.skip_conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=[1, 1], padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.decoder_conv5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, ac_ie_spec, cmp_spec_ac_2_1, cmp_spec_ac_2_2, cmp_spec_ac_2_3, cmp_spec_ac_2_4):
        S_d_in_1 = ac_ie_spec
        S_d_out_1 = self.decoder_conv1(S_d_in_1)

        S_d_in_2 = S_d_out_1 + self.skip_conv1(cmp_spec_ac_2_4)
        S_d_out_2 = self.decoder_conv2(S_d_in_2)

        S_d_in_3 = S_d_out_2 + self.skip_conv2(cmp_spec_ac_2_3)
        S_d_out_3 = self.decoder_conv3(S_d_in_3)

        S_d_in_4 = S_d_out_3 + self.skip_conv3(cmp_spec_ac_2_2)
        S_d_out_4 = self.decoder_conv4(S_d_in_4)

        S_d_in_5 = S_d_out_4 + self.skip_conv4(cmp_spec_ac_2_1)
        S_d_out_5 = self.decoder_conv5(S_d_in_5)

        return S_d_out_5





class IEReNN(nn.Module):
    def __init__(self, channel_amp=9):
        super(IEReNN, self).__init__()

        # self.ie_conv0_0 = nn.Sequential(
        #     nn.Conv2d(channel_amp, 128, kernel_size=[5, 5], padding=(2, 2)),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )

        self.renn_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.renn_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.renn_conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.renn_conv4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=[5, 5], padding=(2, 2)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )


    def forward(self, Sie):
        S_out1 = self.renn_conv1(Sie)
        S_out2 = self.renn_conv2(S_out1)
        S_out3 = self.renn_conv3(S_out2)
        S_out4 = self.renn_conv4(S_out3)

        return S_out4



class DC_SE_Model(nn.Module):

    def __init__(
            self,
            win_len=400,
            win_inc=100,
            fft_len=512,
            win_type='hanning',
            channel_amp=8,

    ):
        super(DC_SE_Model, self).__init__()
        self.num_blocks = 3
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True

        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='real', fix=fix)
        self.ie_stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='real', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='real', fix=fix)

        self.ac_tsbs = ACFeatureEmbedding(input_dim=self.feat_dim,
                                          channel_amp=channel_amp
                                          )

        self.ie_tsbs = IEFeatureEmbedding(input_dim=self.feat_dim,
                                          channel_amp=channel_amp
                                          )

        self.IEReNN = IEReNN(channel_amp=128)
        self.SEN = SENN()


    def forward(self, input_ac, input_ie):

        amp_spec_ac, phase_spec_ac = self.stft(input_ac)
        amp_spec_ie, phase_spec_ie = self.ie_stft(input_ie)


        cmp_spec_ac = torch.unsqueeze(amp_spec_ac, 1)

        cmp_spec_ie = torch.unsqueeze(amp_spec_ie, 1)

        cmp_spec_ac_2_1, cmp_spec_ac_2_2, cmp_spec_ac_2_3, cmp_spec_ac_2_4 = self.ac_tsbs(cmp_spec_ac)
        cmp_spec_ie_2 = self.ie_tsbs(cmp_spec_ie)

        cmp_spec_ie_recon = self.IEReNN(cmp_spec_ie_2)
        squ_spec_ie_recon = torch.squeeze(cmp_spec_ie_recon, 1)
        # SE branch
        ac_ie_spec = torch.cat([cmp_spec_ac_2_4, cmp_spec_ie_2], 1)

        spec = self.SEN(ac_ie_spec, cmp_spec_ac_2_1, cmp_spec_ac_2_2, cmp_spec_ac_2_3, cmp_spec_ac_2_4)
        
        spec = torch.squeeze(spec, 1)

        est_spec = amp_spec_ac * spec

        est_wav = self.istft(est_spec, phase_spec_ac)
        est_wav = torch.squeeze(est_wav, 1)


        return est_spec, est_wav, squ_spec_ie_recon

    def loss(self, est, labels, ie_est, ie_labels):
        '''
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        '''

        b, d, t = est.size()
        gth_amp_spec, gth_phase_spec = self.stft(labels)
        est_amp_spec = est

        ie_gth_amp_est, ie_gth_phase_spec = self.ie_stft(ie_labels)
        ie_est_amp_spec = ie_est

        air_amp_loss = F.mse_loss(gth_amp_spec, est_amp_spec) * d
        log_air_amp_loss = torch.log10(air_amp_loss)

        ie_amp_loss = F.mse_loss(ie_gth_amp_est, ie_est_amp_spec) * d
        log_ie_amp_loss = torch.log10(ie_amp_loss)

        all_loss = log_air_amp_loss + log_ie_amp_loss

        return all_loss, air_amp_loss, ie_amp_loss


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data









