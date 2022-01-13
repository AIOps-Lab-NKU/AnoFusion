# AnoFusion

### Robust Multimodal Failure Detection for Cloud-Native Systems

Anofusion is an unsupervised failure detection model for service instances. It applies GTN, GAT and GRU to learn the correlation of heterogeneous multimodal data as well as capture the normal pattern of service instances to detect failures. To the best of our knowledge, we are among the first to identify the importance of explore the correlation of multimodal data (metrics, logs, and traces), and combine the monitoring data of the three modalities for service instance failure detection.

### Dataset

**Dataset1** is Generic AIOps Atlas4 (GAIA) dataset from CloudWise (https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA collects multimodal data from MicroSS, a business simulation system which contains a two-dimensional code login scenario. The provider injects faults to simulate the anomalies in the real world system, such as regular user behavior and incorrect system operation. 

**Dataset2** is collected in a cloud-native system owned by a large commercial bank supporting hundreds of millions of users with hundreds of service instances.

### Demo Usage

Download MicroSS data, such as a microservice instance mobservice2, the data folder saved are:

> |--data
>
> ​	|--mobservice2_2021-07-01_2021-07-15.csv
>
> ​	|--mobservice2_2021-07-15_2021-07-31.csv
>
> ​	|--mobservice2_stru.csv, mobservice2_temp.csv
>
> ​	|--trace_table_mobservice2_2021-csv 07.csv

cd utils

sh train.sh



