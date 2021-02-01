# Devign - Implementation

In this repository, we provide lightweight implementation of [Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks](https://arxiv.org/pdf/1909.03496.pdf). 

### Requirements
1. Python=3.6 
2. Pytorch==1.4.0
3. [Deep Graph Library](https://www.dgl.ai/)

### Usage
```shell
python main.py \
      --dataset <name_of_the_dataset> \
      --input_dir <directory_of_the_input>;
```

### Datset
The `input_dir` should contain three json files namely
1. `train_GGNNinput.json`
2. `valid_GGNNinput.json`
3. `test_GGNNinput.json`

Each json file should contain a list of json object of the following structure 
```shell
{
  'node_features': <A list of features representing every nodes in the graph>,
  'graph': <A list of edges>
  'target': <0 or 1 representing the vulnerability>
}
```

* Let's assume `n` nodes in the graph are indexed as `0` to `n-1`. The length of `node_features` list should be `n`. Each feature vector should be 100 elements long. Thus the `node_features` list should be a 2D list of shape `(n, 100)`.
  
* The length of `graph` list should be the number of the edges. Each edge should be represented as a three element tuple `[source, edge_type, destination]`. Where the `source` and `destinations` are indices of corresponding node in `node_features` list. Edge types should be from `0` to `max_edge_types`. 

## Note 
1. In this implementation, we followed Devign's paper. We could **NOT** recreate the result in the original paper though.

## Reference
[1] Zhou, Yaqin, et al. "Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks." arXiv preprint arXiv:1909.03496 (2019).
