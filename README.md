# Slot-aware Multi-objective Reranking via Dynamic Weighting

Slot-aware Multi-objective Reranking via Dynamic Weighting (SMR-DW) is modified on [CMR](https://github.com/lyingCS/Controllable-Multi-Objective-Reranking) 

## Requirements

+ Ubuntu 20.04 or later (64-bit)
+ GPU support requires a CUDA®-enabled card
+ For NVIDIA GPUs, the r455 driver must be installed

For wheel installation:
+ Python 3.8
+ pip 19.0 or later

## Quick Started

Our experimental environment is Ubuntu20.04(necessary)+Python3.8(necessary)+CUDA11.4+TensorFlow1.15.5.

#### Create virtual environment(optional)

```
pip install --user virtualenv
~/.local/bin/virtualenv -p python3 ./venv
source venv/bin/activate
```

#### Pull code from github

```
git clone https://github.com/selous123/SMR-DW.git
cd SMR-DW
```

#### Decompress evaluator checkpoint

For facilitate the training of the generator, we adopt the pre-trained CMR_evaluator as our evaluator which is provided by [CMR](https://github.com/lyingCS/Controllable-Multi-Objective-Reranking). We first need to decompress it.

```
tar -xzvf ./model/save_model_ad/10/*.tar.gz -C ./model/save_model_ad/10/
```

#### Run example

Run re-ranker

```
python run_reranker.py --setting_path example/config/ad/slmr_generator_setting.json
```

Model parameters can be set by using a config file, and specify its file path at `--setting_path`, e.g., `python run_ranker.py --setting_path config`. The config files for the different models can be found in `example/config`. Moreover, model parameters can also be directly set from the command line.

**For more information please refer to [LibRerank_README.md](./LibRerank_README.md)**

## Citation

Please cite our paper if you use this repository.

```
@inproceedings{tao2024smrdw,
  title={Slot-aware Multi-objective Reranking via Dynamic Weighting},
  author={Tao Zhang, Mengting Xu and Luwei Yang}
}
```
