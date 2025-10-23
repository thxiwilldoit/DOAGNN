# DOAGNN for FJSP

This is the code for our paper titled: 'Dual Operation Aggregation Graph Neural Networks for Solving Flexible Job-Shop Scheduling Problem with Reinforcement Learning', which has been accepted by WWW2025 (The ACM Web Conference 2025). We encourage everyone to use this code and cite our paper:

```
@inproceedings{zhao2025fjsp,
  title={Dual Operation Aggregation Graph Neural Networks for Solving Flexible Job-Shop Scheduling Problem with Reinforcement Learning},
  author={Peng Zhao, You Zhou, Di Wang, Zhiguang Cao, Yubin Xiao, Xuan Wu, Yuanshu Li, Hongjia Liu, Wei Du, Yuan Jiang, Liupu Wang},
  booktitle={Proceedings of the ACM Web Conference 2025},
  year={2025}
}
```

## Apologize!
We sincerely apologize for mistakenly providing the debug version of `doagnn.py` earlier, which resulted in excessively long inference times. The correct version of `doagnn.py` was uploaded on 10/23/2025.


### Requirements
python $=$ 3.8.17<br>
torch $=$ 1.11.0+cu113<br>
torchaudio $=$ 0.11.0+cu113<br>
torchmetrics $=$ 1.4.0<br>
torchvision $=$ 0.12.0+cu113<br>
matplotlib $=$ 3.7.2<br>
gym $=$ 0.13.0<br>
numpy $=$ 1.24.4<br>
pynvml $=$ 11.5.0<br>
openpyxl $=$ 3.0.10<br>
pandas $=$ 2.0.3<br>

### Introduction
`save`  is used to store the trained weights, Gantt charts, and result data. It can also save the results obtained from testing along with the corresponding Gantt charts.

`train.py` is used for training.

`test.py` is used for evaluating the models. The weights required for testing should be retrieved from the `save` and placed in the `model`.

`config.json` is used to control the parameters within the model.

`model` is used to store the weights that need to be tested. We have included three sets of weights trained on FJSP instances with a size of 10 $\times$ 5.

`fjsp_env` maps the scheduling states of the Flexible Job-Shop Scheduling Problem (FJSP) to the environmental information in the Markov Decision Process (MDP). You can add it to the environment by modifying the `__init__.py` file in the basic Gym library. The example environment location is:" ./anaconda3/envs/env_name/lib/python3.8/site-packages/gym/envs/classic_control/". To add the example code, use:"from gym.envs.classic_control.fjsp_env import FJSPEnv".

### Motivation
We propose a dual operation aggregation graph neural network for solving FJSP. Specifically, we decouple the disjunctive graph into two distinct graphs, reducing graph density and clarifying relationships between machines and operations, thus enabling more effective aggregation and understanding by neural networks. We develop two distinct graph aggregation methods to minimize the influence of non-critical machine and operation nodes on decision-making while enhancing the model's ability to account for long-term benefits. The FJSP environment used in the code is modified from the environment proposed by Song <sup><a href="#ref1">1</a></sup>. We showcase the construction of the corresponding solutions obtained by DOAGNN, along with the Gantt chart, in the folder `MK_result_from_Table_1`.

1. <p name = "ref1"> Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning(https://ieeexplore.ieee.org/document/9826438). <em>IEEE Transactions on Industrial Informatics</em>, 2022.</p>
