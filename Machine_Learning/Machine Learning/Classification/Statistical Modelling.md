# Statistical Modelling

# Naïve Bayes Classifier

Based on statistics, in particular on the Bayes' theorem. It considers the contribution of all the attributes, assuming that each attribute is **independent** from the others, *given the class*. This is a very strong assumption, that is rarely verified, but the method works nevertheless.

The probabilities are estimated with the frequencies

The **Bayes' theorem** affirm:

Given a hypothesis *H* and an evidence *E* that bears on that hypothesis

$$Pr(H|E) = \frac{Pr(E|H)Pr(H)}{Pr(E)}$$

The hypothesis is the class, say *c*, the evidence is the tuple of values of the element to be classified.

We can split the evidence into pieces, one per attribute, and, if they are independent (that's why *naïve*) inside each class:

$$Pr(c|E) = \frac{\prod_i Pr(E_i|c) \cdot Pr(c)}{Pr(E)} $$

## The method

- Compute the conditional probabilities from examples
- Apply the teorem
- The denominator is the same for all the classes, and it's eliminated by the normalization step

## Problem

If a value *v* of attribute *d* never appears in the elements of class *c*, then $Pr(d=v|c) = 0$. This makes the probability of the class for that evidence drops to zero. It is a common case, in particular when there is lots of attributes with lots of values.

## Values not represented in a class - Laplace Smoothing

- $\alpha$: Smoothing parameter, typical value is 1
- $af_{d=v_i,c}$: Absolute frequency of value $v_i$ in attribute *d* over class *c*
- *V*: number of distinct value in attribute $V_i$ over the dataset
- $af_c$: Absolute frequency of class *c* in the dataset

The **Smoothed frequency** is

$$sf_{d=v_i,c} = \frac{af_{d=v_i,c} + \alpha}{af_c + \alpha V}$$

With $\alpha = 0$ we obtain the standard, unsmoothed formula

Higher values of $\alpha$ give more importance to the prior probabilities for the values of *d* w.r.t. the evidence given by the examples

## Missing values

They do not affect the model.

Test instances:

- The calculation of the likelihood simply omits this attribute
- The likelihood will be higher for all the classes, but this is compensated by the normalization

Train instances:

- The record is simply not included in the frequency counts for that attribute
- The descriptive statistics are based on the number of values that occur, rather than on the number of instances

## Numerical values

The method based on frequencies is inapplicable. We assume that the values have a *Gaussian* distribution.

Instead of the fraction of counts, we compute from the example the mean $\mu_c$ and the variance $\sigma_c$ of the values for each numeric attribute **inside each class**

For a given attribute and a given class, the distribution is supposed to be

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \Longrightarrow f(d = x|c) = \frac{1}{\sqrt{2\pi}\sigma_c}e^{-\frac{(x-\mu_c)^2}{2\sigma_c^2}} $$

Then we can compute the likelihood of the class using this probability for the numerical attributes

## Summary

If the condition of independence is violated, for example if one column is the copy of another, then the probability of that feature is weighted more, in the example it's doubled.

If the gaussian distribution of the numerical value is violated, we could use the standard probability estimation for the appropriate distribution, if known, or use estimation procedures

# Linear Classification with the Perceptron

Often called also **artificial neuron**, is in practice a linear combination of weighted inputs.

It could be used to separate examples of two classes, for datasets with numeric attributes.

It learn a *hyperplane* such that all the positives lay on one side and all the negatives on the other

The hyperplane is described by a set of weights $w_0,...,w_D$in a linear equation on the data attributes $e_0,...,e_D$, where the fictitious attribute $e_0 = 1$ is added to allow an hyperplane  that does not pass through the origin

There are either **none** or **infinite** such hyperplanes

$$w_0 * e_0 + w_1 * e_1 + ... + w_D * e_D\begin{cases} > 0 \Rightarrow positive \\
< 0 \Rightarrow negative
\end{cases}$$

```python
set all weights to 0
while there are examples incorrectly classified do
	for each training instances e do
		if e is incorrectly classified do
			if class of e is positive then
				add the e data vector to the vector of weights
			else 
				subtract the e data vector to the vector of weights

```

Each change of weights moves the hyperplane toward the misclassified instance. The result of the equation is increased by $e_0^2 + ... + e_D^2$ if the class of *e* is positive, therefore the result will be **less negative** or, possibly even **positive**. 

The corrections are incremental, and can interfere with previous updates. The algorithm converges if the dataset is **linearly separable**, otherwise it doesn't terminate, so it is necessary to set an upper bound to the iterations

# Support Vector Machines

New efficient separability of non-linear functions that use **kernel functions**, based on *optimization* rather than *greedy search*

### Maximum Margin Hyperplane

The MMH gives the greatest separation between the classes. Finding this and the *support vector*, the subset of points that are sufficient to define the MMH, belongs to the well known class of *constrained quadratic optimization* problems

### Soft Margin Support Vectors classifiers

It's quite common that a separating hyperplane does not exist

- Find an hyperplane which **almost** separate the classes
- disregard examples which generate a very narrow margin

Greater robustness to individual observation and better classification of most of the training observations

### Non-linear class boundaries

The non linearity of boundaries can be overcome with a non-linear mapping. The data are mapped into a *feature space*, that can have a greater number of dimensions, such that a linear boundary in that space can correspond to a non-linear boundary in the original space.

### The *kernel trick*

Some kernel functions allows to do the computation directly in the input space, without switching to the feature space, avoiding an increase of complexity. They are Linear, Polynomial, Exponential, Tanh. The rule of thumb is to start with simpler functions and then try with more complex if necessary 

### Summary

- Learning is in general slower than simpler methods, such as Decision Tree
- Tuning is necessary for some parameters
- Not affected by local minima
- It can be very accurate thanks to the complex boundaries that can be obtained
- No *curse of dimensionality* because it don't use any notion of distance
- It doesn't provide probability estimate, but can produce a confidence score based on the distance of an example from the separation hyperplane

# Neural Networks

Arrange many perceptron-like elements in a hierarchical structure. Another way to overcome the limit of linear decision boundary

A **neuron** is a signal processor with **threshold**. The signal transmission from one neuron to another is **weighted**, and those weights changes over time, also due to **learning**.

The signal transmitted are modeled as real numbers, and the thresholds are functions as Sigmoid, Arctangent

### Sigmoid

Maps reals into ]0, 1[, it's continuous, differentiable, non-linear:

$$S(x) = \frac1{1 + e^{-x}}$$

## Importance of non-linearity

if a function is linear $f(x + \eta) = f(x) + f(\eta)$, so if $\eta$  is generated by noise, it's completely transferred to the output.

If the function is not linear in general $f(x + \eta) \neq f(x) + f(\eta)$

## Feed-Forward multi-layered network

Inputs feed an **input layer**, with one input node for each dimension in the training set

Input layer feed, with weights, an **hidden layer**, with a number of nodes that is a parameter of the network.

Hidden layer feed, with weights, an **output layer**, with a number of nodes that depend on the number of classes in the domain:

- 1 node if there are two classes
- 1 node per class in the other cases (one hot encoding)

The output of each node is $v_k = g(\sum_{i=0}^Dw_{k,i}*e_i)$ where $g(\cdot)$ is the transfer function of the node, e.g. the sigmoid.

Each node of one layer is connected to all the nodes of the following layer. *Feed-Forward* because it flows only in this direction.

## Training the neural network

```python
set all weights to random values
while termination condition is not satisfied do
	for each training instances e do
		1 - feed the network with e and compute the output nn(e)
		2 - compute the weight corrections for nn(e) - e_out
		3 - propagate back the weight corrections
```

The examples must repeatedly feed the network. The weights encode the knowledge given by the supervised examples, but it is not easily understandable. Convergence is not guaranteed.

Important issues:

- computing the weights corrections
- preparation of the training example: standardize the attributes to have zero mean and unit variance
- termination condition

## Computing the error

$$E(\bold{w}) = \frac12(y - Transfer(\bold{w}, \bold{x}))^2$$

We have to move towards a (local) minimum of the error following the **gradient →** compute the partial derivatives of the error as a function of the weights. For example:

$$sgm(x) = \frac1{1 + e^{-x}} \\
\frac{d}{dx}sgm(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = (1 - sgm(x))sgm(x)$$

The weights is changed subtracting the partial derivative multiplied by a **learning rate** constant, that influence the converging speed and the precision

$$w_{ij} \leftarrow w_{ij} - \lambda \frac{\partial E(\bold{w})}{\partial w_{ij}}$$

## Training algorithm

```python
set all weights to random values
while termination condition is not satisfied do
	for each training instances e do
		1 - feed the network with e and compute the output nn(e)
		2 - compute the error prediction at output layer nn(e) - e_out
		3 - compute derivatives and weight corrections for output layer
		4 - compute derivatives and weight corrections for hidden layer
```

## Learning Mode

### Stochastic

The gradient is computed for each data point, and the weight are updates at each forward propagation

It reduces the chance of getting stuck in a local minimum, because it introduces some noise in the gradient descend process

### Batch

Compute the error and the gradient over a batch of data. The updating is performed in the direction of the average error

### Repetition

A learning round over all the samples is called **epoch**. In general after each epoch the network classification capability is improved

### Stop Criteria

- All the weight updates in the epoch have been small
- The classification error rate goes below a predefined target
- A timeout condition is reached

### Risk

- Local minima are possible
- Overfitting is possible, if the network is too complex w.r.t. the complexity of the decision problem

### Regularization

Techniques to improve the generalization capabilities of a model, modifying the **loss function**

# K Nearest Neighbors Classifier

Here the model is the entire training set. The predictions are made by computing the similarity between the new sample and each training instance, and then picks the K entries in the database which are closest to the new data point, and the class is decided with a majority vote.

The main parameters are the number of neighbor to check end the metric used to compute the distance, for example the *Mahalanobis* has good performance.