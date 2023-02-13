# Uncertainty Estimation for Quantized Deep Learning Models: Comparative Study

## Problem Statement
The rise in the popularity of edge computing throws new demands to computing platforms with regards to performance and energy efficiency. In safety critical and real world applications, the guarantees for decision making while using quantized deep learning model is vital and is required to evaluate the model uncertainty. The model must be aware of the fact that it will ultimately be quantized, so that it can perform all weight adjustments accordingly during training, to yield higher accurate predictions. QAT outperforms post training quantization methods and is used in this research.

This R&D aims to find and compare the probability distribution functions fn, uncertainty in model predictions, where n ∈ N, representing different uncertainty estimation methods. The function composes input data x and either non-quantized model parameters θ or quantized model parameters θ′. In order to evaluate the impact of quantization from the perspective of prediction reliability, the probability distribution functions f1(x, θ) and f1(x, θ′) are compared. In addition, this project compares the functions f1(x, θ′), f2(x, θ′), f3(x, θ′).......fn(x, θ′) to evaluate the uncertainty estimation methods for the quantized deep learning models.

## Research Questions
What are the differences in uncertainty estimates of quantized deep learning models with different uncertainty estimation methods for classification, regression and segmentation tasks?
• Taxonomy of uncertainty estimation methods in deep learning with special
focus on single pass uncertainty estimation method
• Literature search on different quantization methods and which are supported
by the DUT
• How does evidential loss function impact the QAT?
