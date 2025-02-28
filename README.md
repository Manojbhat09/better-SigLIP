### AdaptiveSigLIP

An enhanced implementation of SigLIP (Sigmoid Loss for Language Image Pre-Training) that improves training efficiency and model performance through adaptive parameters and curriculum learning.

#### Key Features

- **Learnable Temperature & Bias**: Automatically optimizes these critical parameters during training instead of using fixed values
- **Curriculum Learning**: Implements progressive training from lower to higher image resolutions for better convergence
- **Comparative Framework**: Includes tools to compare curriculum vs. standard training approaches
- **Performance Metrics**: Comprehensive evaluation metrics including AUC, convergence speed, and accuracy
- **Flexible Dataset Support**: Works with multiple datasets (COCO, Flickr8k, CIFAR10, etc.) for easy experimentation

#### Why AdaptiveSigLIP?

SigLIP has shown promising results for vision-language models, but requires careful tuning of temperature and bias parameters. This implementation removes the need for manual tuning by making these parameters learnable, while also implementing curriculum learning strategies that have been shown to improve model convergence and final performance.

Our experiments demonstrate [X]% improvement in convergence speed and [Y]% better final accuracy compared to standard SigLIP training.
