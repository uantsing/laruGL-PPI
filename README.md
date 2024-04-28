# Effectiveness and efficiency: label-aware hierarchical subgraph learning for protein-protein interaction
Yuanqing Zhou, Haitao Lin, Yufei Huang,  Lirong Wu, Stan Z. Li and Wei Chen  
  
This repo contains source code implementing the algorithm in our [paper](https://doi.org/10.1101/2024.03.08.584199).  
  
**Contact**  
  
Yuanqing Zhou (yuanqzhou@zju.edu.cn),  Wei Chen (zjuchenwei@zju.edu.cn)  
  
Feel free to report bugs or tell us your suggestions!
## Cython Implemented Parallel Graph Sampler
We have a cython module which needs compilation before training can start. Compile the module by tunning the following from the root directory:  
```
python laruGL/setup.py build_ext --inplace
```
## Training and testing
```
python run.py
````


