# RawNet2 ASVspoof 2021 baseline

------

The code in this repository serves as one of the baselines of the ASVspoof 2021 challenge, using an end-to-end method that uses a model based on the RawNet2 topology as described [here](https://arxiv.org/abs/2011.01108).

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/asvspoof-challenge/2021.git
$ cd 2021/LA/Baseline-RawNet2/
$ conda create --name rawnet_anti_spoofing python=3.6.10
$ conda activate rawnet_anti_spoofing
$ conda install pytorch=1.4.0 -c pytorch
$ pip install -r requirements.txt
```

## Experiments

### Dataset
Our model for the deepfake (DF) track is trained on the logical access (LA) train  partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Training
To train the model run:
```
python main.py --track=DF --loss=CCE   --lr=0.0001 --batch_size=32
```

### Testing

To test your own model on the ASVspoof 2021 DF evaluation set:

```
python main.py --track=DF --loss=CCE --is_eval --eval --model_path='/path/to/your/your_best_model.pth' --eval_output='eval_CM_scores.txt'
```

We also provide a pre-trained model which follows a Mel-scale distribution of the sinc filters at the input layer, which can be downloaded from [here](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip). To use it you can run: 
```
python main.py --track=DF --loss=CCE --is_eval --eval --model_path='/path/to/your/pre_trained_DF_model.pth' --eval_output='pre_trained_eval_CM_scores.txt'
```

If you would like to compute scores on the development set of ASVspoof 2019 simply run:

```
python main.py --track=DF --loss=CCE --eval --model_path='/path/to/your/best_model.pth' --eval_output='dev_CM_scores.txt'
```

```
