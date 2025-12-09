# VGG Implementation (Simonyan et al., 2015)

This project attempts to replicate the architecture described in the paper
***Very Deep Convolutional Networks for Large-Scale Image Recognition*** by **Simonyan & Zisserman (2015)**.



## Overview

The VGG paper presents several model variants (A, B, C, D, E), all built around the core idea that instead of using large convolution kernels, we can stack multiple **3×3 convolutions** to achieve the same receptive field more efficiently.

This project implements VGG with certain modifications to make training easier on smaller datasets and limited hardware.



## Architecture

![Architecture](static/diagram.png)
*Architecture of the implemented VGG-Small*

The implemented architecture includes the following changes compared to the original:

1. **Fewer Channels:**
   Later blocks use **256 channels instead of 512**, reducing model size and improving training speed.

2. **Smaller Classifier:**
   Instead of the original **Linear → Linear → Output** classifier, this model uses a simplified **Linear → Output** design.

### Additional Architectures Implemented (But Not Trained Fully)

1. VGG-11
2. VGG-13
3. VGG-16
4. VGG-19

These were implemented to test convergence but not trained to completion.

The VGG-Small architecture was created specifically to avoid overfitting and to make training manageable under limited compute.



## Dataset

The original VGG models were trained on the **ImageNet** dataset.
Due to compute constraints, this implementation uses **CIFAR-10**, consisting of **60,000 images across 10 classes**.

A batch size of **256** was used for training.



## Training

Training was performed on an **NVIDIA T4 GPU** on Google Colab.

* Initial learning rate: **0.01**, reduced twice during training
* Loss function: **CrossEntropyLoss**
* Optimizer: **Adam**
* Training duration: **~4 hours total**
* Weights were saved and reused whenever the compute session ended

After **30 epochs**, the model achieved:

* **94.6% accuracy** on the training set
* **86.2% accuracy** on the test set



## Usage

### Clone the repository

```bash
git clone https://github.com/Vidit-01/VGG-2015-implementation.git
cd VGG-2015-implementation
pip install -r requirements.txt
```

### Train a model

```bash
python train.py models/vgg-small.py
```

### Override default hyperparameters

```bash
python train.py models/vgg-small.py --epochs 60
```

### Arguments Summary

| Argument        | Description                                            | Default      |
| --------------- | ------------------------------------------------------ | ------------ |
| `model_path`    | Path to the `.py` file containing the model definition | **Required** |
| `--epochs`      | Number of training epochs                              | `30`         |
| `--bs`          | Batch size                                             | `64`         |
| `--lr`          | Learning rate                                          | `0.01`       |
| `--num_classes` | Number of output classes                               | `200`        |
| `--workers`     | Number of dataloader workers                           | `2`          |
| `--out`         | Directory to save results                              | `results/`   |
| `--device`      | `cuda`, `cpu`, or `auto`                               | `auto`       |

## How I Got Here

I began by reading the VGG paper, and after implementing AlexNet, it was clear that training VGG would be significantly more challenging because of its higher parameter count and computational cost. I initially attempted training VGG-16 on TinyImageNet (100k images, 200 classes) using my [AlexNet pipeline](https://github.com/Vidit-01/alexnet-2012-implementation/tree/main), but training was extremely slow—around **18–20 minutes per epoch**.

To improve performance, I switched to Kaggle for access to two T4 GPUs and removed CPU-heavy image transformations. Despite this, after 10 epochs, accuracy was still **0.05**, indicating the model had learned nothing.

I then rewrote the dataset loading and training pipeline directly in Jupyter, reducing training time to about **10 minutes per epoch**. Still, the model trained too slowly, so I switched to **CIFAR-10**, a smaller dataset with fewer classes.

Even then, full VGG-16 remained slow, so I created **VGG-Small**, a lighter version with fewer channels and a reduced classifier. This model trained in **under 5 minutes per epoch** and converged reliably.

Afterward, I implemented **VGG-11, VGG-13, VGG-16, and VGG-19** to verify basic functionality, though they have not been fully trained yet.


## Citation
Simonyan, K., & Zisserman, A. (2015).
**Very Deep Convolutional Networks for Large-Scale Image Recognition.**
[Paper](https://arxiv.org/abs/1409.1556)

