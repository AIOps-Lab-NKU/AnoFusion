# AnoFusion

### Robust Multimodal Failure Detection for Microservice Systems

Anofusion is an unsupervised failure detection model for service instances. It applies GTN, GAT and GRU to learn the correlation of heterogeneous multimodal data as well as capture the normal pattern of service instances to detect failures. To the best of our knowledge, we are among the first to identify the importance of explore the correlation of multimodal data (metrics, logs, and traces), and combine the monitoring data of the three modalities for service instance failure detection.

### Dataset

**Dataset1** is Generic AIOps Atlas4 (GAIA) dataset from CloudWise (https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA collects multimodal data from MicroSS, a business simulation system which contains a two-dimensional code login scenario. The provider injects faults to simulate the anomalies in the real world system, such as regular user behavior and incorrect system operation. 

**Dataset2** is collected in a cloud-native system owned by a large commercial bank supporting hundreds of millions of users with hundreds of service instances.

### Install
Use pip to install these packages
```
numpy==1.19.2
pandas==1.1.5
scipy==1.5.4
scikit-learn==0.24.2
torch==1.6.0
tqdm==4.62.3
pyyaml==6.0
```

### Demo Usage

1. Download MicroSS data, such as a microservice instance mobservice2, the data folder saved are:
```
File Tree:
└─data
   ├── mobservice2_2021-07-01_2021-07-15.csv
   ├── mobservice2_2021-07-15_2021-07-31.csv
   ├── mobservice2_stru.csv
   ├── mobservice2_temp.csv
   └── trace_table_mobservice2_2021-csv 07.csv
```

2. Preprocess dataset and enter the `utils` folder, run `python3 generate_channels.py`, and get the input of AnoFusion.

3. Edit the configuration file `config.py`
4. Taking `mobservice2` as an example, the model training is completed by executing `python3 main.py --mode train --service_s mobservice2`. We have encapsulated this command into the `train.sh` and can directly execute `sh train.sh`.
5. Taking `mobservice2` as an example, the model evaluation is completed by executing `python3 main.py --mode eval --service_s mobservice2`.

### Package Description
```
File Tree:
.
├── README.md
├── config.py (State the name of the microservice and the range of timestamps.)
├──data (Store the initial monitoring data of the three modalities.)
    ├── mobservice2_2021-07-01_2021-07-15.csv
    ├── mobservice2_2021-07-15_2021-07-31.csv
    ├── mobservice2_stru.csv
    ├── mobservice2_temp.csv
    └── trace_table_mobservice2_2021-csv 07.csv
├── labeled_service (The failure labels of microservices, which are used in the model evaluation phase.）
    └── mobservice2.csv
├── model (The main components of the model.)
    ├── AnoFusion.py
    ├── GAT.py
    ├── GATGRU.py
    ├── GTblock.py
    ├── GTlayer.py
    └── MyDataset.py
├── serialize (Assist in the serialization of logs and traces.)
    ├── log_to_sequence.py
    └── trace_to_sequence.py
└── utils (The main execution files.）
    ├── generate_channels.py (Process the raw multimodal data into an input form for the AnoFusion.)
    ├── main.py
    ├── train.sh (Execution file for training phase.)
    └── eval.sh (Execution file for evaluation phase.)
```

### Citing AnoFusion
AnoFusion paper is published in KDD 2023. If you use AnoFusion, we would appreciate citations to the following paper:

`Robust Multimodal Failure Detection for Microservice Systems.`

> By Chenyu Zhao, Minghua Ma, Shenglin Zhang, Dan Pei, et.al.

BibTex:
```
@inproceedings{zhao2023robust,
  author       = {Zhao, Chenyu and Ma, Minghua and Zhong, Zhenyu and Zhang, Shenglin and Tan, Zhiyuan and Xiong, Xiao and Yu, LuLu and Feng, Jiayi and Sun, Yongqian and Zhang, Yuzhi and others},
  title        = {Robust Multimodal Failure Detection for Microservice Systems},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6-10, 2023},
  pages        = {5639--5649},
  publisher    = {{ACM}},
  year         = {2023},
  doi          = {10.1145/3580305.3599902}
}
```
