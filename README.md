# Neurocoder
Code of Neurocoder paper  
ICML version: https://proceedings.mlr.press/v162/le22b.html  
Code Ref:
- Continual Learning tasks: https://github.com/GT-RIPL/Continual-Learning-Benchmark
- Other tasks: TBU


# Setup  
```
pip install -r requirements.txt
```
Install other packages if possible


# Continual Learning Tasks

```
cd cl

```
Run baseline MLP
```
./scripts/split_MNIST_incremental_domain.sh  mlp
```
Run baseline Neurocoder
```
./scripts/split_MNIST_incremental_domain.sh  nsa
```
Notes: 
- Results are logged in outputs/ 
- Choose CL backbone by modifying scripts/split_MNIST_incremental_domain.sh  
- Core model code is in models/nsa.py
