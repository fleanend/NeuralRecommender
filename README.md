# NeuralRecommender


[![PyPI](https://img.shields.io/pypi/v/neuralrecommender.svg)](https://pypi.org/project/neuralrecommender/)

NeuralRecommender is a PyTorch implementation of a number of recommendation algorithms using neural networks.

As of now it implements:
- [GLocal-K: Global and Local Kernels for Recommender Systems](https://arxiv.org/pdf/2108.12184.pdf)



## Installation
Install from `pip`:
```
pip install neuralrecommender
```


## Quickstart
Fitting a model on the MovieLens 100k dataset:
```python
import numpy as np
import ml_metrics
from neuralrecommender.glocalk import GlocalK

# Load the MovieLens 100k dataset.
def load_data_100k(): 
... # code in example notebook
return 
n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k()

# Instantiate and train the model
recommender = GlocalK()
metrics = recommender.fit(train_r)

# Recommend for all users
res = recommender.predict(np.arange(n_u))

# Evaluate the recommendations
k=50
ground_truth = np.argsort(-test_r, axis=0)[:k,:].T.tolist()
recommended = np.argsort(-res, axis=0)[:k,:].T.tolist()
random = np.random.randint(0,n_m,(n_u, k)).T.tolist()

ml_metrics.mapk(ground_truth, random, k=k)
ml_metrics.mapk(ground_truth, recommended, k=k)
```

## References
1. [GLocal-K: Global and Local Kernels for Recommender Systems](https://arxiv.org/pdf/2108.12184.pdf)
2. [GLocal-K official implementation](https://github.com/usydnlp/Glocal_K)
3. Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4), 1-19.
