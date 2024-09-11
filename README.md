


# Introduction
EarSpeech is an earphone-based speech enhancement system that exploits in-ear channel speech as the complementary modality to enable airborne speech enhancement. The key idea of EarSpeech is that in-ear speech is less sensitive to ambient noise and exhibits a high correlation with airborne speech which is sensitive to ambient noise. The goal of EarSpeech is to fuse the in-ear speech to improve the quality and intelligibility of airborne speech. Throughout extensive experiments, EarSpeech achieves an average improvement ratio of 27.23% and 13.92% in terms of PESQ and STOI, respectively, and significantly improves SI-SDR by 8.91 dB. Benefiting from data augmentation, EarSpeech can achieve comparable performance with a small-scale dataset that is 40 times less than the original dataset. In addition, EarSpeech presents a higher generalization of different users, speech content, and language types, respectively, as well as a stronger robustness in the real world. More technical details and surprising results can be found in our paper which is published on ACM IMWUT/Ubicomp 2024 [paper](https://dl.acm.org/doi/10.1145/3678594).

If you think our work is helpful to you, please cite our paper:

  @article{10.1145/3678594,
    author = {Han, Feiyu and Yang, Panlong and Zuo, You and Shang, Fei and Xu, Fenglei and Li, Xiang-Yang},
    title = {EarSpeech: Exploring In-Ear Occlusion Effect on Earphones for Data-efficient Airborne Speech Enhancement},
    year = {2024},
    issue_date = {August 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {8},
    number = {3},
    url = {https://doi.org/10.1145/3678594},
    doi = {10.1145/3678594},
    month = {sep},
    articleno = {104},
    numpages = {30}
  }


# 1. Quick Reproduction

The model of EarSpeech and the pre-trained model are released in [model](https://github.com/EarSpeech/earspeech.github.io/tree/main/model)  

**References**: huyanxin's [phasen](https://github.com/huyanxin/phasen/tree/master)

# 2. Audio Demo of EarSpeech

Here, we release some audio demo samples to demonstrate the performance of EarSpeech. 

The structure of the folder is shown as follows:

<div align="left"> 	<img src="./floder_structure.png" width="30%"> </div>


- "SNR_-5dB_0dB", "SNR_0dB_5dB", and "SNR_5dB_10dB" represent the SNR of noisy airborne speech ranges from [-5, 0] dB, [0, 5] dB, and [5, 10] dB, respectively.  "Chinese_samples" and "English_samples" represent the speech in Chinese and English, respectively.
- "Read_world_study" represents the speech collected in noisy real-world environments ( noise SPLs of the two environments are 72.19 dB and 75.27 dB, respectively).


We first show the comparison between  **(1) clean airborne speech (reference)**, **(2) corresponding in-ear speech**, **(3) noisy airborne speech (mixing clean speech with various noise)**, and **(4) enhanced airborne speech**

## 2.1 SNR_-5dB_0_dB

### 2.1.1 Chinese samples [Audio files are in [audioDemo](./SNR_-5dB_0dB/Chinese_samples/) ]

<div align="left"> 	<img src="./SNR_-5dB_0dB/Chinese_samples/time.png" width="100%"> </div>


<div align="left"> 	<img src="./SNR_-5dB_0dB/Chinese_samples/time-frequency.png" width="100%"> </div>



### 2.1.2 English samples [Audio files are in [audioDemo](./SNR_-5dB_0dB/English_samples/) ]

<div align="left"> 	<img src="./SNR_-5dB_0dB/English_samples/time.png" width="100%"> </div>


<div align="left"> 	<img src="./SNR_-5dB_0dB/English_samples/time-frequency.png" width="100%"> </div>





## 2.2 SNR_0dB_5_dB

### 2.2.1 Chinese samples [Audio files are in [audioDemo](./SNR_0dB_5dB/Chinese_samples/) ]

<div align="left"> 	<img src="./SNR_0dB_5dB/Chinese_samples/time.png" width="100%"> </div>


<div align="left"> 	<img src="./SNR_0dB_5dB/Chinese_samples/time-frequency.png" width="100%"> </div>



### 2.2.2 English samples [Audio files are in [audioDemo](./SNR_0dB_5dB/English_samples/) ]

<div align="left"> 	<img src="./SNR_0dB_5dB/English_samples/time.png" width="100%"> </div>


<div align="left"> 	<img src="./SNR_0dB_5dB/English_samples/time-frequency.png" width="100%"> </div>



## 2.3 SNR_5dB_10_dB

### 2.3.1 Chinese samples [Audio files are in [audioDemo](./SNR_5dB_10dB/Chinese_samples/) ]

<div align="left"> 	<img src="./SNR_5dB_10dB/Chinese_samples/time.png" width="100%"> </div>


<div align="left"> 	<img src="./SNR_5dB_10dB/Chinese_samples/time-frequency.png" width="100%"> </div>



### 2.3.2 English samples [Audio files are in [audioDemo](./SNR_-5dB_0dB/English_samples/) ]

<div align="left"> 	<img src="./SNR_5dB_10dB/English_samples/time.png" width="100%"> </div>

<div align="left"> 	<img src="./SNR_5dB_10dB/English_samples/time-frequency.png" width="100%"> </div>



## 2.4 Real_world_study



### 2.3.1 Env1_Noise_SPL_72.19dB [Audio files are in [audioDemo](./Real_world_study/Env1_Noise_SPL_72.19dB/) ]

<div align="left"> 	<img src="./Real_world_study/Env1_Noise_SPL_72.19dB/time.png" width="100%"> </div>

<div align="left"> 	<img src="./Real_world_study/Env1_Noise_SPL_72.19dB/time-frequency.png" width="100%"> </div>

### 2.3.2 Env2_Noise_SPL_75.27dB [Audio files are in [audioDemo](./Real_world_study/Env2_Noise_SPL_75.27dB/) ]

<div align="left"> 	<img src="./Real_world_study/Env2_Noise_SPL_75.27dB/time.png" width="100%"> </div>

<div align="left"> 	<img src="./Real_world_study/Env2_Noise_SPL_75.27dB/time-frequency.png" width="100%"> </div>


# 3. Contact Information
fyhan@mail.ustc.edu.cn



