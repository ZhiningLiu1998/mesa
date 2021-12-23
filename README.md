<h1 align="center"> MESA: Meta-sampler for imbalanced learning </h1>

<p align="center">
  <!-- <img src="https://img.shields.io/badge/ZhiningLiu1998-MESA-orange">
  <img src="https://img.shields.io/github/stars/ZhiningLiu1998/mesa">
  <img src="https://img.shields.io/github/forks/ZhiningLiu1998/mesa">
  <img src="https://img.shields.io/github/issues/ZhiningLiu1998/mesa">  
  <img src="https://img.shields.io/github/license/ZhiningLiu1998/mesa"> -->
  <a href="https://github.com/ZhiningLiu1998/mesa">
    <img src="https://img.shields.io/badge/ZhiningLiu1998-MESA-orange">
  </a>
  <a href="https://github.com/ZhiningLiu1998/mesa/stargazers">
    <img src="https://img.shields.io/github/stars/ZhiningLiu1998/mesa">
  </a>
  <a href="https://github.com/ZhiningLiu1998/mesa/network/members">
    <img src="https://img.shields.io/github/forks/ZhiningLiu1998/mesa">
  </a>
  <a href="https://github.com/ZhiningLiu1998/mesa/issues">
    <img src="https://img.shields.io/github/issues/ZhiningLiu1998/mesa">
  </a>
  <a href="https://github.com/ZhiningLiu1998/mesa/graphs/traffic">
    <img src="https://visitor-badge.glitch.me/badge?page_id=ZhiningLiu1998.mesa">
  </a>
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<a href="https://github.com/ZhiningLiu1998/mesa#contributors-"><img src="https://img.shields.io/badge/all_contributors-1-orange.svg"></a>
<!-- ALL-CONTRIBUTORS-BADGE:END -->
  <a href="https://github.com/ZhiningLiu1998/mesa/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/ZhiningLiu1998/mesa">
  </a>
</p>

<h3 align="center"> MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler (NeurIPS 2020)
</h3>

<h3 align="center">
Links: 
<a href="https://papers.nips.cc/paper/2020/file/a64bd53139f71961c5c31a9af03d775e-Paper.pdf">Paper</a> | 
<a href="https://arxiv.org/pdf/2010.08830.pdf">PDF with Appendix</a> | 
<a href="https://studio.slideslive.com/web_recorder/share/20201020T134559Z__NeurIPS_posters__17343__mesa-effective-ensemble-imbal?s=d3745afc-cfcf-4d60-9f34-63d3d811b55f">Video</a> | 
<a href="https://arxiv.org/abs/2010.08830">arXiv</a> | 
<a href="https://zhuanlan.zhihu.com/p/268539195">Zhihu/知乎</a>
</h3>

**MESA is a ***meta-learning-based ensemble learning framework*** for solving class-imbalanced learning problems. It is a task-agnostic general-purpose solution that is able to boost most of the existing machine learning models' performance on imbalanced data.**

<!-- > **NOTE:** The paper will be available through [arXiv](https://arxiv.org/) in a few days. We will provide a link to the .pdf file ASAP. -->

# Cite Us

**If you find this repository helpful in your work or research, we would greatly appreciate citations to the following paper:**

```
@inproceedings{liu2020mesa,
    title={MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler},
    author={Liu, Zhining and Wei, Pengfei and Jiang, Jing and Cao, Wei and Bian, Jiang and Chang, Yi},
    booktitle={Conference on Neural Information Processing Systems},
    year={2020},
}
```

# Table of Contents

- [Cite Us](#cite-us)
- [Table of Contents](#table-of-contents)
- [Background](#background)
  - [About MESA](#about-mesa)
  - [Pros and Cons of MESA](#pros-and-cons-of-mesa)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Running main.py](#running-mainpy)
  - [Running mesa-example.ipynb](#running-mesa-exampleipynb)
- [Visualization and Results](#visualization-and-results)
  - [From mesa-example.ipynb](#from-mesa-exampleipynb)
    - [Class distribution of Mammography dataset](#class-distribution-of-mammography-dataset)
    - [Visualize the meta-training process](#visualize-the-meta-training-process)
    - [Comparison with baseline methods](#comparison-with-baseline-methods)
  - [Other results](#other-results)
    - [Dataset description](#dataset-description)
    - [Comparisons of MESA with under-sampling-based EIL methods](#comparisons-of-mesa-with-under-sampling-based-eil-methods)
    - [Comparisons of MESA with over-sampling-based EIL methods](#comparisons-of-mesa-with-over-sampling-based-eil-methods)
    - [Comparisons of MESA with resampling-based EIL methods](#comparisons-of-mesa-with-resampling-based-eil-methods)
- [Miscellaneous](#miscellaneous)
- [References](#references)
  - [Contributors ✨](#contributors-)


# Background

## About MESA

We introduce a novel ensemble imbalanced learning (EIL) framework named MESA. It adaptively resamples the training set in iterations to get multiple classifiers and forms a cascade ensemble model. MESA directly learns a parameterized sampling strategy (i.e., meta-sampler) from data to optimize the final metric beyond following random heuristics. It consists of three parts: ***meta sampling*** as well as ***ensemble training*** to build ensemble classifiers, and ***meta-training*** to optimize the meta-sampler. 

The figure below gives an overview of the MESA framework. 

![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/framework.png)

## Pros and Cons of MESA

Here are some personal thoughts on the advantages and disadvantages of MESA. More discussions are welcome!

**Pros:**
- &#x1F34E; *Wide compatiblilty.*   
We decoupled the model-training and meta-training process in MESA, making it compatible with most of the existing machine learning models.
- &#x1F34E; *High data efficiency.*  
MESA performs strictly balanced under-sampling to train each base-learner in the ensemble. This makes it more data-efficient than other methods, especially on highly skewed data sets.
- &#x1F34E; *Good performance.*  
The sampling strategy is optimized for better final generalization performance, we expect this can provide us with a better ensemble model.
- &#x1F34E; *Transferability.*  
We use only task-agnostic meta-information during meta-training, which means that a meta-sampler can be directly used in unseen new tasks, thereby greatly reducing the computational cost brought about by meta-training.

**Cons:**
- &#x1F34F; *Meta-training cost.*  
Meta-training repeats the ensemble training process multiple times, which can be costly in practice (By shrinking the dataset used in meta-training, the computational cost can be reduced at the cost of minor performance loss).
- &#x1F34F; *Need to set aside a separate validation set for training.*  
The meta-state is formed by computing the error distribution on both the training and validation sets.
- &#x1F34F; *Possible unstable performance on small datasets.*  
Small datasets may cause the obtained error distribution statistics to be inaccurate/unstable, which will interfere with the meta-training process.

# Requirements
**Main dependencies:**
- [Python](https://www.python.org/) (>=3.5)
- [PyTorch](https://pytorch.org/) (=1.0.0)
- [Gym](https://gym.openai.com/) (>=0.17.3)
- [pandas](https://pandas.pydata.org/) (>=0.23.4)
- [numpy](https://numpy.org/) (>=1.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.20.1)
- [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html) (=0.5.0, optional, for baseline methods)

To install requirements, run:

```Shell
pip install -r requirements.txt
```

> **NOTE**: this implementation requires an old version of PyTorch (v1.0.0).
> You may want to start a new conda environment to run our code. The step-by-step guide is as follows (using torch-cpu for an example):
> - `conda create --name mesa python=3.7.11`
> - `conda activate mesa`
> - `conda install pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 cpuonly -c pytorch`
> - `pip install -r requirements.txt`
> 
> These commands should help you to get ready for running mesa. If you have any further questions, please feel free to open an issue or drop me an email.

# Usage

A typical usage example:

```python
# load dataset & prepare environment
args = parser.parse_args()
rater = Rater(args.metric)
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(args.dataset)
base_estimator = DecisionTreeClassifier()

# meta-training
mesa = Mesa(
    args=args, 
    base_estimator=base_estimator, 
    n_estimators=10)
mesa.meta_fit(X_train, y_train, X_valid, y_valid, X_test, y_test)

# ensemble training
mesa.fit(X_train, y_train, X_valid, y_valid)

# evaluate
y_pred_test = mesa.predict_proba(X_test)[:, 1]
score = rater.score(y_test, y_pred_test)
```

## Running [main.py](https://github.com/ZhiningLiu1998/mesa/blob/master/main.py)

Here is an example:

```powershell
python main.py --dataset Mammo --meta_verbose 10 --update_steps 1000
```

You can get help with arguments by running:

```powershell
python main.py --help
```

```
optional arguments:
  # Soft Actor-critic Arguments
  -h, --help            show this help message and exit
  --env-name ENV_NAME
  --policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --eval EVAL           Evaluates a policy every 10 episode (default:
                        True)
  --gamma G             discount factor for reward (default: 0.99)
  --tau G               target smoothing coefficient(τ) (default: 0.01)
  --lr G                learning rate (default: 0.001)
  --lr_decay_steps N    step_size of StepLR learning rate decay scheduler
                        (default: 10)
  --lr_decay_gamma N    gamma of StepLR learning rate decay scheduler
                        (default: 0.99)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.1)
  --automatic_entropy_tuning G
                        Automaically adjust α (default: False)
  --seed N              random seed (default: None)
  --batch_size N        batch size (default: 64)
  --hidden_size N       hidden size (default: 50)
  --updates_per_step N  model updates per simulator step (default: 1)
  --update_steps N      maximum number of steps (default: 1000)
  --start_steps N       Steps sampling random actions (default: 500)
  --target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size N       size of replay buffer (default: 1000)

  # Mesa Arguments
  --cuda                run on CUDA (default: False)
  --dataset N           the dataset used for meta-training (default: Mammo)
  --metric N            the metric used for evaluate (default: aucprc)
  --reward_coefficient N
  --num_bins N          number of bins (default: 5). state-size = 2 *
                        num_bins.
  --sigma N             sigma of the Gaussian function used in meta-sampling
                        (default: 0.2)
  --max_estimators N    maximum number of base estimators in each meta-
                        training episode (default: 10)
  --meta_verbose N      number of episodes between verbose outputs. If 'full'
                        print log for each base estimator (default: 10)
  --meta_verbose_mean_episodes N
                        number of episodes used for compute latest mean score
                        in verbose outputs.
  --verbose N           enable verbose when ensemble fit (default: False)
  --random_state N      random_state (default: None)
  --train_ir N          imbalance ratio of the training set after meta-
                        sampling (default: 1)
  --train_ratio N       the ratio of the data used in meta-training. set
                        train_ratio<1 to use a random subset for meta-training
                        (default: 1)
```

## Running [mesa-example.ipynb](https://github.com/ZhiningLiu1998/mesa/blob/master/mesa-example.ipynb)

We include a highly imbalanced dataset [Mammography](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.datasets.fetch_datasets.html#imblearn.datasets.fetch_datasets) (#majority class instances = 10,923, #minority class instances = 260, imbalance ratio = 42.012) and its variants with flip label noise for quick testing and visualization of MESA and other baselines. 
You can use [mesa-example.ipynb](https://github.com/ZhiningLiu1998/mesa/blob/master/mesa-example.ipynb) to quickly:
- conduct a comparative experiment
- visualize the meta-training process of MESA
- visualize the experimental results of MESA and other baselines

**Please check [mesa-example.ipynb](https://github.com/ZhiningLiu1998/mesa/blob/master/mesa-example.ipynb) for more details.**

# Visualization and Results

## From [mesa-example.ipynb](https://github.com/ZhiningLiu1998/mesa/blob/master/mesa-example.ipynb)

### Class distribution of Mammography dataset
![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/class-distribution.png)

### Visualize the meta-training process
<!-- ![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/meta-training-process.png) -->
<p align="center">
  <img src="https://github.com/ZhiningLiu1998/figures/blob/master/mesa/meta-training-process.png" />
</p>

### Comparison with baseline methods
![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/result.png)

## Other results

### Dataset description

![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/datasets.png)

### Comparisons of MESA with under-sampling-based EIL methods

![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/comp-USEIL.png)

### Comparisons of MESA with over-sampling-based EIL methods

![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/comp-OSEIL.png)

### Comparisons of MESA with resampling-based EIL methods

![image](https://github.com/ZhiningLiu1998/figures/blob/master/mesa/comp-resample.png)



# Miscellaneous

**Check out our previous work [Self-paced Ensemble](https://github.com/ZhiningLiu1998/self-paced-ensemble) (ICDE 2020).  
It is a simple heuristic-based method, but being very fast and works reasonably well.**

**This repository contains:**
- Implementation of MESA
- Implementation of 7 ensemble imbalanced learning baselines
  - `SMOTEBoost` [1]
  - `SMOTEBagging` [2]
  - `RAMOBoost` [3]
  - `RUSBoost` [4]
  - `UnderBagging` [5]
  - `BalanceCascade` [6]
  - `SelfPacedEnsemble` [7]
- Implementation of 11 resampling imbalanced learning baselines [8]

> **NOTE:** The implementations of the above baseline methods are based on [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) and [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn). 

# References

| #   | Reference |
|-----|-------|
| [1] | N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer, Smoteboost: Improving prediction of the minority class in boosting. in European conference on principles of data mining and knowledge discovery. Springer, 2003, pp. 107–119|
| [2] | S. Wang and X. Yao, Diversity analysis on imbalanced data sets by using ensemble models. in 2009 IEEE Symposium on Computational Intelligence and Data Mining. IEEE, 2009, pp. 324–331.|
| [3] | Sheng Chen, Haibo He, and Edwardo A Garcia. 2010. RAMOBoost: ranked minority oversampling in boosting. IEEE Transactions on Neural Networks 21, 10 (2010), 1624–1642.|
| [4] | C. Seiffert, T. M. Khoshgoftaar, J. Van Hulse, and A. Napolitano, Rusboost: A hybrid approach to alleviating class imbalance. IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans, vol. 40, no. 1, pp. 185–197, 2010.|
| [5] | R. Barandela, R. M. Valdovinos, and J. S. Sanchez, New applications´ of ensembles of classifiers. Pattern Analysis & Applications, vol. 6, no. 3, pp. 245–256, 2003.|
| [6] | X.-Y. Liu, J. Wu, and Z.-H. Zhou, Exploratory undersampling for class-imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539–550, 2009. |
| [7] | Zhining Liu, Wei Cao, Zhifeng Gao, Jiang Bian, Hechang Chen, Yi Chang, and Tie-Yan Liu. 2019. Self-paced Ensemble for Highly Imbalanced Massive Data Classification. 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020, pp. 841-852.
| [8] | Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas. Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17):1–5, 2017. |
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://zhiningliu.com"><img src="https://avatars.githubusercontent.com/u/26108487?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zhining Liu</b></sub></a><br /><a href="#ideas-ZhiningLiu1998" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ZhiningLiu1998/mesa/commits?author=ZhiningLiu1998" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
