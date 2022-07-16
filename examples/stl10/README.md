# STL-10 Representation Learning with Alignment and Uniformity Losses

## Requirements
```
Python >= 3.6
torch >= 1.5.0
torchvision
```

## Getting Started
+ Training an encoder:
  ```sh
  python main.py
  ```
  
  You may use `--gpus` to specify multiple GPUs to use, e.g., `--gpus 1 3`.

  See [main.py](./main.py) for more command-line arguments.

+ Evaluating an encoder:
  ```sh
  python linear_eval.py [PATH_TO_ENCODER]
  ```
  
  You may use `--gpu` to specify the GPU to use, e.g., `--gpu 3`.

  See [linear_eval.py](./linear_eval.py) for more command-line arguments.

## Reference Validation Accuracy
83.19% using 4 GPUs with default options:
+ AlexNet-variant encoder architecture.
+ Loss: `L_align(alpha=2) + L_uniform(t=2)`.
+ Classification on penultimate layer (fc7) activations.

With 8 GPUs, @sachit-menon kindly reported 83.39%.

