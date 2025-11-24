# ChaosNexus

ChaosNexus: A Foundation Model for Universal Chaotic System Forecasting with Multi-scale Representations

>"Accurately forecasting chaotic systems, prevalent in domains such as weather prediction and fluid dynamics, remains a significant scientific challenge. The inherent sensitivity of these systems to initial conditions, coupled with a scarcity of observational data, severely constrains traditional modeling approaches. Since these models are typically trained for a specific system, they lack the generalization capacity necessary for real-world applications, which demand robust zero-shot or few-shot forecasting on novel or data-limited scenarios.
To overcome this generalization barrier, we propose ChaosNexus, a foundation model pre-trained on a diverse corpus of chaotic dynamics. ChaosNexus employs a novel multi-scale architecture named ScaleFormer augmented with Mixture-of-Experts layers, to capture both universal patterns and system-specific behaviors. The model demonstrates state-of-the-art zero-shot generalization across both synthetic and real-world benchmarks. On a large-scale testbed comprising over 9,000 synthetic chaotic systems, it improves the fidelity of long-term attractor statistics by more than 40\% compared to the leading baseline. This robust performance extends to real-world applications with exceptional data efficiency. For instance, in 5-day global weather forecasting, ChaosNexus achieves a competitive zero-shot mean error below 1°C—a result that further improves with few-shot fine-tuning. Moreover, experiments on the scaling behavior of ChaosNexus provide a guiding principle for scientific foundation models: cross-system generalization stems from the diversity of training systems, rather than sheer data volume."

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