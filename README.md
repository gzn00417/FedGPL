# Federated Graph Prompt Learning

Graph Prompt Learning (GPL) has attracted increasing attention for eliciting the generalizability of the Pre-trained Graph Model (PGM) on various downstream graph tasks. However, privacy concerns on PGM and data simultaneously arise when fine-tuning an available PGM on privately owned domain-specific graphs. To this end, we propose a privacy-preserving framework, Federated Graph Prompt Learning (FedGPL), for fine-tuning personalized graph models among participants with heterogeneous graph tasks and data. First, we construct a split framework for end-to-end fine-tuning with joint preservation of PGM and data privacy. This framework isolates PGM and graph data against potential unauthorized access and communicates by differentially privatized graph representations and gradients. Then, to overcome task and data heterogeneity, we propose a hierarchical aggregation algorithm to distill task- and data- specific beneficial knowledge in a disentangled manner. Besides, we devise a graph prompt to softly insert denoized homogeneous structures to further mitigate task and data heterogeneity. Extensive experiments demonstrate the effectiveness of FedGPL in terms of accuracy against various baseline FL frameworks and GPL methods under privacy constraints.

## Running experiments

* To finetune on Cora with FedGPL and ratio=0.1:
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
