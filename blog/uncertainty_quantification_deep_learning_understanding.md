# Uncertainty estimation for deep learning models

**Objective:**  Probabilistic learning with deep neural networks for uncertainty estimation <br>
**Motivation:**  Traditional deep learning model tend to propagate biases from training and are often susceptible to failures on brand new OOD data, meaning it is assumed that the training data is drawn from the same distribution as test data<br>
**Result:** Model that can reliably and quickly estimate uncertainty in input data as well as model predictions<br>
**Conclusion:** Though the acuracy of model is slightly lower, the evidence for when to trust the model must be available<br>


# Deterministic Neural Networks
Determinsitic neural networks model the expectation of our target, which results in the point estimate of prediction <br>
Input data x, Target y -> Prediction E[y] <br>

**Classification**
* Targets: Discrete - Natural numbers, y $\in$ {1,...,k}
* Given an input data x and discrete target classes y, the model predictions are E[y]

**Regression**
* Targets: Continuous - Real numbers, y $\in$ $\mathbb{R}$
* Given an input data x and target real number y, the model predictions are E[y]

The drawbacks are point estimate lacks the understanding of how spread or uncertain the predictions are. So insted of just predicting the expectation of y, the model must also estimate the variance of y. <br>
Input data x, Tragte y -> Prediction _Var_[y] <br>


# Likelihood
The probability distribution of the targets

**Classification**
* Targets: Discrete - Natural numbers, y $\in$ {1,...,k}
* Probability distribution over discrete class categories
* Activation function outputs distribution parameters p (probabilities)
* Uses softmax activation function to satisfy two constraints: each probability outputs had to be greater than 0 and sum all class probabilities are normalized to 1.
* Uses negative log likelihood of predicted _multinomial_ distribution to match the ground truth category distribution (cross entropy loss)
* y ~ Categorical(p)
    * y is class label
    * Categorical is likelihood function
    * p is distribution parameters (probabilities)

**Regression**
* Targets: Continuous - Real numbers, y $\in$ $\mathbb{R}$
* Probability distribution over the entire real number line
* Activation function outputs distribution parameters $\mu$ (mean), $\sigma$ (standard deviation)
* Uses exponential activation function to satisfy one constraint: standard deviation must be strictly positive
* Uses negative log likelihood of predicted _normal_ distribution to match the ground truth normal distribution (Gaussian loss)
* y ~ Normal( $\mu$, $\sigma^2$)
    * y is target label
    * Normal is likelihood function
    * $\mu$, $\sigma^2$ are distribution parameters

|  | Classification | Regression |
|---|---|---|
|Targets| y $\in$ {1,...,k} | y $\in$ $\mathbb{R}$ | 
|Likelihood | y ~ Categorical(p) | y ~ Normal( $\mu$, $\sigma^2$) |
|Parameters | p = {p1,...,pk} | ( $\mu$, $\sigma^2$)|
|Constraints | $\sum_{i} p_{i}$ = 1;  $p_{i}$ > 0| $\mu \in \mathbb{R}$; $\sigma$ > 0|
|Loss function | Cross Entropy $$-\sum_{i=1}^{K} y_{i} logp_{i}$$ | Negtaive Log-Likelihood $$-log(N(y \| \mu, \sigma^2))$$ |

The drawbacks of likelihood is it's unreliable outputs, if the input is unlike anything during training. <br>
Example: Feeding a ship image in a Cat, Dog trained model. Though the probabilities of likelihood be anything, it cannot be trusted and confidence value is required. This is beacuse of the uncertainty and their types are:

| Aleatoric | Epistemic |
|---|---|
|Data uncertainty| Model uncertainty |
|Describes the confidence in input data | Describes the confidence of prediction |
|High when input data is noisy | High when missing training data | 
|Cannot be reduced by adding more data | Can be reduced by adding more data| 
|Can be learned directly using NN with likelihood estimation techniques | Can be solved using Bayesian NN instead of deterministic NN| 
|For regression problem, giving data x to model with likelihood estimation technique provides ( $\mu$, $\sigma^2$)| For the same problem, giving data x to model with likelihood estimation techniques and ensemble (many independently trained instances of the model), multiple ( $\mu$, $\sigma^2$) are produced|
|$\sigma^2$ is the data uncertainty | The variance of $\mu$ from multiple instances of ensemble is model uncertainty. If var[ $\mu$] is lesser, the model is more confident and vice versa| 

| Distribution parameter | Type of uncertainty|
|---|---|
|Low variance in $\mu$ and $\sigma^2$| Low uncertainty |
|High variance in $\sigma^2$| Data uncertainty|
|High variance in $\mu$ | Model uncertainty|


| Deterministic NN | Bayesian NN |
|---|---|
|Single number representation for every weight| Probability distribution representation for every weight|
|Learn fixed set of weights $$W$$ | Learn posterior over weights $$P(W\|X,Y)$$ Learns posterior with sampling methods such as dropout, ensembles|
|Given a set of weights, passing in one input to the model multiple times will yield the same fixed output | sampling of the weight distribution is performed for every single weight resulting in slightly different outputs for different iterations|


# Evidential Deep learning
Instead of sampling to learn posteriors such as in ensemble methods, the distribution of the likelihood parameters can be learnt directly and are called as evidential deep learning. This distribution is a higher order distribution of the likelihood distributions parameters and evidential deep learning directly estimates both aleatoric and epistemic uncertainty.

**Classification**
* y ~ Categorical(p)
    * y is class label
    * Categorical is likelihood function
    * p is distribution parameters (probabilities)
* The learning of the likelihood distribution parameters for classification task in evidential deep learning are $p$ 
* p ~ Dirichlet( $\alpha$)
    * p is distribution parameters 
    * Dirichlet is evidential prior
    * $\alpha$ is model parameters
* Data x ----> Neural Network ----> Classification ( $\alpha$)

**Regression**
* y ~ Normal( $\mu$, $\sigma^2$)
    * y is target label
    * Normal is likelihood function
    * $\mu$, $\sigma^2$ are distribution parameters
* The learning of the likelihood distribution parameters for regression task in evidential deep learning are $\mu$, $\sigma^2$ 
* $\mu$, $\sigma^2$ ~ NormalInvGamma( $\gamma, \nu, \alpha, \beta$)
    * $\mu$, $\sigma^2$ are distribution parameters
    * NormalInvGamma is Evidential prior
    * $\gamma, \nu, \alpha, \beta$ are model parameters
* Data x ----> Neural Network ----> Regression ( $\gamma, \nu, \alpha, \beta$)

|  | Classification | Regression |
|---|---|---|
|Targets| y $\in$ {1,...,k} | y $\in$ $\mathbb{R}$ | 
|Likelihood | y ~ Categorical(p) | y ~ Normal( $\mu$, $\sigma^2$) |
|Evidential distribution| p ~ Dirichlet( $\alpha$) | $\mu$, $\sigma^2$ ~ NormalInvGamma( $\gamma, \nu, \alpha, \beta$)|

Normal Inverse Gamma and Dirichlet are the selected evidential distributions, because they are called conjugate priors. Their form makes the analytical computation for loss tractable.

|  | Likelihood estimation| Bayesian NN |Evidential NN|
|---|---|---|---|
|Prior placed over| Data | Weights | Likelihood|
|Weights are | Deterministic | Stochastic | Deterministic|
|Fast (no sampling)| Yes | No | Yes|
|Captures epistemic uncertainty| No | Yes | Yes |


