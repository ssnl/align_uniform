# Alignment and Uniformity Metrics for Representation Learning

This repository provides a PyTorch implementation of the *alignment* and *uniformity* metrics for unsupervised representation learning. These metrics are proposed in ***Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere***.

+ [arXiv Paper](https://arxiv.org/abs/2005.10242)
+ [Project Page](https://ssnl.github.io/hypersphere)

These metrics/losses are useful for:
1. (as metrics) quantifying encoder feature distribution properties,
2. (as losses) directly training the encoder.

Requirements:
+ PyTorch >= 1.5.0

## Documentation

Thanks to their simple forms, these losses are implemented in [just a few lines of code in `align_uniform/__init__.py`](align_uniform/__init__.py#L4-L9):
```py
# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
```

After `import align_uniform`, you can access them with
```py
align_uniform.align_loss(x, y)

align_uniform.uniform_loss(x)
```

## Examples

We provide the following [examples](./examples) to perform unsupervised representation learning using these two losses:
+ [STL-10](examples/stl10)
+ [ImageNet and ImageNet-100 with a MoCo Variant](https://github.com/SsnL/moco/tree/align_uniform)

## Citation

Tongzhou Wang, Phillip Isola. "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere". *International Conference on Machine Learning*. 2020.

```
@article{wang2020hypersphere,
  title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author={Wang, Tongzhou and Isola, Phillip},
  journal={arXiv preprint arXiv:2005.10242},
  year={2020}
}
```

## Questions

For questions about the code provided in this repository, please open an GitHub issue.

For questions about the paper, please contact Tongzhou Wang (`tongzhou _AT_ mit _DOT_ edu`).
