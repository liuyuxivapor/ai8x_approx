# Neural Network Model Based Signal Processing Approximation on AIoT Processors 

<p align="middle">
    <a href="https://www.mdpi.com/2079-9292/14/6/1064"><img src="https://img.shields.io/badge/MDPI-2079--9292%2F14%2F6%2F1064-blue"/></a>
</p>
 
## Overview

Approximate computing of at least five certain signal processing such as FFT and FIR will be implemented in AI accelerators, Cortex-M4F processors, and DSP, where the efficiency of each implementation method will be comparatively evaluated. The performance of approximations will be improved by optimizing the deep learning network structure and other methods and reduce device power consumption. It is expected to achieve performance and power consumption comparable to MCU or even DSP on MAX78000.


## Logs & Todo

| Signal                  | NAS | Ai8x              | Evaluation        | CMSIS-DSP | DSP(?) |
|:-----------------------:|:-------:|:-----------------:|:-----------------:|:---------:|:------:|
| Fingerprint Recognition | /       | ⚪                 | ⚪                 | /         | /      | 
| FFT                     | ⚪       |  ⚪  |   ⚪                | ⚪         |        |    ?     |
| FIR                     | ⚪       | ⚪                   |     ⚪              | ⚪         |     ?   | 
| DCT                     | ⚪       | ⚪  |⚪   | ⚪         |     ?   | 
| Biquad IIR              |     ⚪    |     ⚪              |     ⚪              |  ⚪         |   ?     |

## Ref Links

[MAXIM MSDK Guide](https://analog-devices-msdk.github.io/msdk/USERGUIDE)

[ai8x-training & synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis)

[MAX78000 user guide](https://www.analog.com/media/en/technical-documentation/user-guides/max78000-user-guide.pdf)

[MaximAI Documentation](https://github.com/MaximIntegratedAI/MaximAI_Documentation)

[CMSIS DSP instruction](https://www.keil.com/pack/doc/CMSIS/DSP/html)

[My Zhihu Column](https://www.zhihu.com/column/c_1701895548897017856)
