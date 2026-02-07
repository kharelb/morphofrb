<h1 align="center">MorphoFRB</h1>

**MorphoFRB** is a Python package and command line tool for morphological classification of **Fast Radio Bursts (FRBs)** using a deep learning (ConvNext) model fine tuned on [**CHIME/FRB Catalog 2**](https://www.chime-frb.ca/catalog2) data set. It supports easy inference from `.npy` files of FRB dynamic spectrum in the frequency window of 400-800 MHz. Training data set typically contained ~160 time samples at ~1 ms cadence, while spectral channels were first bin-averaged down to 256 before bilinear interpolation reshaped the dynamic spectra to ConvNext-friendly 224 x 224 tensors.

The current model has not been evaluated on FRBs from other radio telescopes, but the package allows on-the-fly fine tuning to adapt the model to new datasets.

We exprimented with various state of the art computer vision models including vision transformers but for current data set ConvNext outperformed them all. 


[![DOI](https://img.shields.io/badge/DOI-10.3847%2FXXXXX-blue)](https://doi.org/10.3847/1538-4357/ae323c)
[![arXiv](https://img.shields.io/badge/arXiv-25XX.XXXXX-B31B1B.svg)](
https://doi.org/10.48550/arXiv.2509.06208
)
## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/kharelb/morphofrb.git
cd morphofrb
pip install .
```

Alternatively, install directly with `pip`:

```bash
pip install morphofrb
```

## Model Weights

- **Hosted on:** Hugging Face
- **Downloaded:** automatically on first inference
- **Cached locally at:** `~/.cache/morphofrb/`
- **Manual setup:** No manual setup required

## Quick Start
### 1. Inferenc (CLI)
#### Classify a single FRB file
```bash
morphofrb-infer --files_path /path/to/file.npy --print_results
```
⚠️ If your path contains spaces, quote it.


#### Classify all files in a directory

```bash
morphofrb-infer --files_path /path/to/npy_directory --save_prediction --out_dir .
```
This will create:
```bash
prediction_YYYYMMDD_HHMMSS.json
```

#### CLI Options
```
--files_path        Path to a .npy file or directory (required)
--state_dict        Path to custom model weights (.pth)
--save_prediction   Save predictions as JSON
--out_dir           Output directory (default: current working directory)
--batch_size        Batch size for inference (default: 32)
--print_results     Print predictions to terminal
```

### 2. Inference (Python API)
```python
from morphofrb.inference import predict

predictions, filenames = predict(
    files_path="/path/to/data",
    print_results=True,
)
```
You can also provide custom weights after you further fine tune:
```python
predictions, filenames = predict(
    files_path="/path/to/data",
    model_state_dict_path="my_model.pth"
)
```

### 3. Fine Tuning (CLI)
#### Dataset Structure
Training and validation directories must follow a class-per-subfolder layout:
```text
train_dir/
├── class0/
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
└── class1/
    ├── sample_101.npy
    ├── sample_102.npy
    └── ...

val_dir/
├── class0/
│   ├── sample_901.npy
│   └── ...
└── class1/
    ├── sample_951.npy
    └── ...
```

Fine tune with default configuration
```bash
morphofrb-finetune --train_dir /path/to/train_dir --val_dir /path/to/val_dir
```

Save outputs to a directory
```bash
morphofrb-finetune \
  --train_dir /path/to/train_dir \
  --val_dir /path/to/val_dir \
  --out_dir /path/to/outputs
```
Use focal loss
```bash
morphofrb-finetune \
  --train_dir /path/to/train_dir \
  --val_dir /path/to/val_dir \
  --focal_gamma 2.0 \
  --out_dir /path/to/outputs
```

Use config file (recommended) for training hyperparameters
```bash
morphofrb-finetune \
  --train_dir /path/to/train_dir \
  --val_dir /path/to/val_dir \
  --config config.json
```

### 4. Fine tuning (Python API)
#### Basic example
```python
from morphofrb.fine_tune import fine_tune
from morphofrb.config import Config

# instantiate config with necessary changes
# if config not instantiated default values retained
# e.g. change batch size to 16 and num epoch 50 and so on...
config = Config(
    batch_size=16,
    num_epochs=50,
    learning_rate=3e-4,
    fine_tune_chime_cattwo=True,
    early_stopping=True,
    patience=5,
)
fine_tune(
    train_dir=Path("/path/to/train_dir"),
    val_dir=Path("/path/to/val_dir"),
    out_dir=Path("./outputs"),
    focal_gamma=2.0,
    config=config,
)
```
## Outputs
Each fine-tuning run creates a timestamped directory inside `out_dir`:
```text
out_dir/
└── 20260206_221530/
    ├── final_trained_model.pth
    ├── trained_model_{epoch}.pth
    └── train_results.csv
```
- `final_trained_model.pth` — model weights corresponding to last epoch
- `trained_model_{epoch}.pth` - model weights corresponding to best validation loss
- `train_results.csv` — per-epoch loss and accuracy

### Configuration (config.json)
Fine-tuning behavior is controlled by a JSON config that maps directly to the Config dataclass.

#### Default configuration fields
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_size` | `int` | `32` | Batch size used during training |
| `num_epochs` | `int` | `500` | Maximum number of training epochs |
| `learning_rate` | `float` | `1e-3` | Adam optimizer learning rate |
| `learning_rate_scheduler` | `bool` | `True` | Enable MultiStepLR scheduler |
| `learning_rate_milestones` | `list[int]` | `[15, 30, 50, ...]` | Epochs at which LR decays |
| `scheduler_gamma` | `float` | `0.1` | LR decay factor |
| `early_stopping` | `bool` | `True` | Enable early stopping |
| `patience` | `int` | `20` | Early stopping patience |
| `weighted_sampling` | `bool` | `True` | Use weighted sampler for imbalance |
| `replacement_sampling` | `bool` | `True` | Sample with replacement |
| `shuffle_train` | `bool` | `False` | Shuffle training data |
| `shuffle_val` | `bool` | `False` | Shuffle validation data |
| `fine_tune_chime_cattwo` | `bool` | `False` | Start from CHIME Cat-2 weights |
| `model_weight` | `str` | `"DEFAULT"` | Base pretrained weight mode |
| `feature_layers_to_unfreeze` | `dict[int, list[int]]` | `{0: [0]}` | ConvNeXt layers to unfreeze |
| `unfreeze_classification_layer` | `bool` | `True` | Unfreeze classifier head |
| `set_workers` | `bool` | `False` | Explicitly set DataLoader workers |
| `num_workers` | `int` | `8` | Number of DataLoader workers |
| `pin_memory` | `bool` | `False` | Enable pinned memory |
| `print_training_details` | `bool` | `True` | Print per-epoch logs |


#### Important JSON note*
JSON keys must be strings. When using:
```json
"feature_layers_to_unfreeze": { "0": [0] }
```
the key `"0"` is converted to `int(0)` internally.

## Model architecture summary 
- ConvNeXt backbone
- Binary classification head
- ImageNet normalization
- Grayscale input expanded to 3 channels

## Citation / acknowledgment
If you find this work useful, please cite:
```bibtex
@article{Kharel_2026,
doi = {10.3847/1538-4357/ae323c},
url = {https://doi.org/10.3847/1538-4357/ae323c},
year = {2026},
month = {feb},
publisher = {The American Astronomical Society},
volume = {998},
number = {1},
pages = {154},
author = {Kharel, Bikash and Fonseca, Emmanuel and Brar, Charanjot and Khan, Afrokk and Mas-Ribas, Lluis and Patil, Swarali Shivraj and Scholz, Paul and Siegel, Seth Robert and Stenning, David C.},
title = {Repeating versus Nonrepeating Fast Radio Bursts: A Deep Learning Approach to Morphological Characterization},
journal = {The Astrophysical Journal},
abstract = {We present a deep learning approach to classify fast radio bursts (FRBs) based purely on morphology as encoded on recorded dynamic spectrum from Canadian Hydrogen Intensity Mapping Experiment (CHIME)/FRB Catalog 2. We implemented transfer learning with a pretrained ConvNext architecture, exploiting its powerful feature extraction ability. ConvNext was adapted to classify dedispersed dynamic spectra (which we treat as images) of the FRBs into one of the two subclasses, i.e., repeater and nonrepeater, based on their various temporal and spectral properties and the relation between the subpulse structures. Additionally, we also used a mathematical model representation of the total intensity data to interpret the deep learning model. Upon fine-tuning the pretrained ConvNext on the FRB spectrograms, we were able to achieve high classification metrics while substantially reducing training time and computing power as compared to training a deep learning model from scratch with random weights and biases without any feature extraction ability. Importantly, our results suggest that the morphological differences between repeating and nonrepeating CHIME events persist in Catalog 2 and the deep-learning model leveraged these differences for classification. The fine-tuned deep-learning model can be used for inference, which enables us to predict whether an FRB’s morphology resembles that of repeaters or nonrepeaters. Such inferences may become increasingly significant when trained on larger datasets that will exist in the near future.}
}
```

## Licence
```md
MIT License © 2026 Bikash Kharel
```



