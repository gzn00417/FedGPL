# Against Multifaceted Graph Heterogeneity via Asymmetric Federated Prompt Learning

Federated Graph Learning (FGL) aims to collaboratively and privately optimize graph models on divergent data for different tasks. A critical challenge in FGL is to enable effective yet efficient federated optimization against multifaceted graph heterogeneity to enhance mutual performance. However, existing FGL works primarily address graph data heterogeneity and perform incapable of graph task heterogeneity. To address the challenge, we propose a Federated Graph Prompt Learning (FedGPL) framework to efficiently enable prompt-based asymmetric graph knowledge transfer between multifaceted heterogeneous federated participants. Generally, we establish a split federated framework to preserve universal and domain-specific graph knowledge, respectively. Moreover, we develop two algorithms to eliminate task and data heterogeneity for advanced federated knowledge preservation. First, a Hierarchical Directed Transfer Aggregator (HiDTA) delivers cross-task beneficial knowledge that is hierarchically distilled according to the directional transferability. Second, a Virtual Prompt Graph (VPG) adaptively generates graph structures to enhance data utility by distinguishing dominant subgraphs and neutralizing redundant ones. We conduct theoretical analyses and extensive experiments to demonstrate the significant accuracy and efficiency effectiveness of FedGPL against multifaceted graph heterogeneity compared to state-of-the-art baselines on large-scale federated graph datasets.

## Running experiments

* To train the FedGPL (VPG and HiDTA by default) on Cora and ratio=0.1:
    ```
    python main.py --algorithm VPG --dataset_name Cora --federated HiDTA --num_classes 7 --ratio 0.1
    ```

## Options for Training

* ```--algorithm:``` Algorithm you want to run. Default: 'ProG'. Options: VPG, 
* ```--data_name``` Name of dataset. Default: 'CiteSeer'. Options: Citeseer, 
* ```--federated:``` Federated algorithm you want to run. Default: HiDTA
* ```--num_classes:``` Number of node classes. Default: 6, for CiteSeer,
* ```--num_users:``` Number of users. Default: 9,
* ```--gnn_type:``` Type of GNN. Default: 'TransformerConv',
* ```--rounds:``` Number of global communication rounds. Default: 10,
* ```--local_epochs:``` Number of local training epochs. Default: 1.
* ```--seed:``` Seed. Default: 2023.

## Data Preprocess
Procedure: 
1. **Feature Reducing using SVD**;
2. **Node/Edge Splitting**: Grouping Nodes/Edges by their Lables. **Note that**, for the Edge_level task, only choose those whose starting point has the same label as the endpoint;
3. **Graph induced**: Build an subgraph for each Node/Edge sample.
* To preprocess the CiteSeer with smallest_size is 100, largest_size is 300 of an induced graph:
    ```
    python data_preprocess.py --data_name CiteSeer --smallest_size 100 --largest_size 300
    ```

## Options for Data Preprocess

* ```--data_name:``` Dataset you want to use. Default: CiteSeer,
* ```--smallest_size:``` Smallest number of nodes a reduced graph has. Default: 100,
* ```--largest_size:``` Largest number of nodes a reduced graph has. Default: 300.
