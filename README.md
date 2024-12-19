# Does GCL Need a Large Number of Negative Samples? Enhancing Graph Contrastive Learning with Effective and Efficient Negative Sampling (AAAI 2025)

The official source code for "Does GCL Need a Large Number of Negative Samples? Enhancing Graph Contrastive Learning with Effective and Efficient Negative Sampling".



Part of code is referenced from [Deep Graph Contrastive Representation Learning](https://github.com/CRIPAC-DIG/GRACE) and [PyGCL: A PyTorch Library for Graph Contrastive Learning](https://github.com/PyGCL/PyGCL))



## Environment Setup

- torch==2.1.0
- torch-geometric==2.5.3
- torch-scatter==2.1.2
- torch-sparse==0.6.18
- scikit-learn==1.2.0
- scipy==1.10.1
- numpy==1.24.3



## Usage

To train and test the model on a specific dataset, use:

```python
bash run.sh <Dataset>  # Example: PubMed, CS, Photo, Computers, Physics, or Wiki-CS.
```







