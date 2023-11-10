# Neural Network Model Based Signal Processing Approximation on AIoT Processors 

## Overview

Approximate computing of at least five certain signal processing such as FFT and FIR will be implemented in AI accelerators, Cortex-M4F processors, and DSP, where the efficiency of each implementation method will be comparatively evaluated. The performance of approximations will be improved by optimizing the deep learning network structure and other methods and reduce device power consumption. It is expected to achieve performance and power consumption comparable to MCU or even DSP on MAX78000.

## Logs & Todo

| **Signal**          | **Pytorch** | **Ai8x** | **CMSIS-DSP** | **DSP** | **Evaluation** | **DVFS(?)** |
|:-------------------:|:-----------:|:--------:|:-------------:|:-------:|:--------------:|:-----------:|
| Fingerprint recognition | /           | ⚪        | /             | /       | ⚪              |  /
| FFT                 | ⚪           |          | ⚪             | ⚪       | ⚪              |             |
| FIR                 | ⚪           |          | ⚪             | ⚪       |                |             |
| DCT                 | ⚪           |  2023.11.09-234801      | ⚪             |         |                |             |
| Linear Interpolate  |             |          |               |         |                |             |

## Ref Links

[MAXIM MSDK Guide](https://analog-devices-msdk.github.io/msdk/USERGUIDE)

[ai8x-training & synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis)

[MAX78000 user guide](https://www.analog.com/media/en/technical-documentation/user-guides/max78000-user-guide.pdf)

[MaximAI Documentation](https://github.com/MaximIntegratedAI/MaximAI_Documentation)

[CMSIS DSP instruction](https://www.keil.com/pack/doc/CMSIS/DSP/html)

[My Zhihu Column](https://www.zhihu.com/column/c_1701895548897017856)
