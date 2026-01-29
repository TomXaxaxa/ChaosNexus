# ChaosNexus

ChaosNexus: A Foundation Model for ODE-based Chaotic System Forecasting with Hierarchical Multi-scale Awareness

>"Foundation models have shown great promise in achieving zero-shot or few-shot forecasting for ODE-based chaotic systems via large-scale pretraining. However, existing architectures often fail to capture the multi-scale temporal structures and distinct spectral characteristics of chaotic dynamics. To address this, we introduce ChaosNexus, a foundation model for chaotic system forecasting underpinned by the proposed ScaleFormer architecture. By processing temporal contexts across hierarchically varying patch sizes, ChaosNexus effectively captures long-range dependencies and preserves high-frequency fluctuations. To address heterogeneity across distinct systems, we integrate Mixture-of-Experts (MoE) layers into each ScaleFormer block and explicitly condition the final forecasts on a learned frequency fingerprint, providing the model with a global spectral view of the system. Extensive evaluations on over 9,000 synthetic systems demonstrate that ChaosNexus achieves superior fidelity in long-term attractor statistics while maintaining competitive point-wise accuracy. Furthermore, in real-world applications, it achieves a remarkable zero-shot mean error below 1°C for 5-day station-based weather forecasting. Codes are available at {https://anonymous.4open.science/r/ChaosNexus-C809}."

# Our Model
![model schematic](data/ChaosNexus.png)

# Installation
## Environment
- Tested OS: Linux
- Python 3.11.0
- PyTorch 2.8.0

# Dataset

Obtain the released major dataset from [[Hugging Face]](https://huggingface.co/datasets/GilpinLab/skew40), and the Weather-5K dataset from [[OneDrive]](https://hkustconnect-my.sharepoint.com/:u:/g/personal/thanad_connect_ust_hk/EZGm7DP0qstElZwafr_U2YoBk5Ryt9rv7P31OqnUBZUPAA?e=5r0wEo). 
Then place the downloaded data in the folder`./data`.

# Pre-trained Models

If you require the pre-trained parameters of ChaosNexus for reproduction or fine-tuning, please download them from [[Dropbox]](https://www.dropbox.com/scl/fo/i2r3ds0emg9sdcqe7vm9q/AHFx8RWPirS6dEUD6lT3Fig?rlkey=6ckgecajs35zbhek6hmpe0fel&st=7unoznyu&dl=0).

# Experiment Reproduction

## 1. Major Dataset Experiments

To reproduce the results on the major dataset, please follow the instructions below for different models.

### Training

**ChaosNexus**
1. Open `./scaleformer/scaleformer/scaleformer.py` and set the variable `MODEL_NAME` to `'Nexus'`.
2. Run the training script:
```bash
bash ./scripts/scaleformer/run_predict_finetune-Nexus.sh
```

**Panda (Baseline)**
1. Open `./scaleformer/scaleformer/scaleformer.py` and set the variable `MODEL_NAME` to `'Panda'`.
2. Run the training script:
```bash
bash ./scripts/scaleformer/run_predict_finetune-Panda.sh
```

**Chronos (Fine-tuning)**
1. Download the public pre-trained weights.
2. Run the fine-tuning script:
```bash
bash ./scripts/chronos/run_finetune.sh
```

### Evaluation

After training is complete, evaluate the models using:
```bash
bash ./scripts/scaleformer/run_eval.sh
```

or evaluate Chronos-SFT using:
```bash
bash ./scripts/chronos/run_eval.sh
```

## 2. Weather Dataset Experiments

For the Weather-5K dataset, additional preprocessing and specific evaluation steps are required.

**Preprocessing**
After downloading the data to `./data`, convert the format using:
```bash
python ./scripts/transform_weather.py
```

**Fine-tuning Configuration**
Before starting the fine-tuning process, you must modify the model initialization logic. Please open `./scripts/train.py`, locate the `Initialize model` section, comment out the currently active code block, and uncomment the alternative block specifically for fine-tuning.

**Inference**
Run the prediction script. **Crucially**, you must ensure the following flags are set to `true` to generate the necessary output files for benchmarking:
```bash
# Ensure these arguments are included in your run command
eval.save_labels=true \
eval.save_predictions=true \
eval.save_contexts=true
```

## 3. Baseline Models Reproduction

To reproduce the results of other baseline models (control group) used for comparison in our paper, we utilize the **Time-Series-Library**. 

Please refer to the official repository: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) for the implementation of these advanced deep time series models. We follow the standard training and evaluation protocols provided in their library to ensure a fair comparison.

**Benchmarking**
Once the predictions are generated, run the benchmark script to obtain the final results:
```bash
python ./scripts/benchmark_weather.py
```

# License
The software in this repository is freely available under MIT license. Please see the license for further details.
