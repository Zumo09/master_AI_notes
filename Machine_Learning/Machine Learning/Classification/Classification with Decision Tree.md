# Classification with Decision Tree

A ***Decision Tree*** is a tree-sharped set of test, that has *inner nodes* and *leaf nodes*.

```python
Inner nodes:
	if test on attribute d of element e then
		execute node'
	else
		execute node''

Leaf node:
	predict class of element e as c''
```

## Learning a Decision Tree - Model Generation

Given a set $\epsilon$ of elements for which the class is known, grow a decision tree as follows:

- if all the elements belong to class $c$ or $\epsilon$ is *small*
    - generate a leaf node with label $c$
- otherwise
    - choose a test based on a single attribute with two or more outcomes
    - make this test the root of a tree with one branch for each outcomes of the test
    - partition $\epsilon$ into subsets corresponding to the outcomes and apply recursively the procedures to the subsets

## Entropy

Given a source *X* with *V* possible values, with probability distribution $P(v_1) = p_1, ..., P(v_V) = p_V$ the entropy of the information source is:

$$H(X) = -\sum_j p_i log_2(p_j)$$

High entropy means that the probabilities are mostly similar → the histogram would be flat

Low entropy means that some symbols have much higher probability → the histogram would have peaks

## Conditional Specific Entropy

Getting insight on attribute Y knowing attribute X

$H(Y|X = v)$ : Entropy of Y considering only the rows such that X = v, it's the weighted average of the conditional specific entropy of Y w.r.t. the values of X

$$H(Y|X) = \sum_j P(X=v_j)*H(Y|X=v_j) $$

It's the average number of buts necessary to transmit the value of Y if both ends of the communication knows the values of X

## Information Gain

$$IG(Y|X) = H(Y) - H(Y|X)$$

If $H(Y|X) < H(Y) \rightarrow IG(Y|X) > 0$   then X provides some insight on Y

Higher IG means that a 2-D contingency table would be more interesting.

## Back to DT generation

In the growing of a Decision Tree *which attribute should we test*?

- Test the attribute which guarantees the maximum IG for the class attribute in the current dataset $\epsilon$

```python
buildTree(dataset e, node p)
	if all the class values of e are c then
		return node p is a leaf, label of p is c
	if no attribute can give a positive information gain in e then
		say that the majority of elements in e has class c
		return node p is a leaf, label of p is c
	find attribute d giving maximum information gain in e
	say that there are V distinct values of d in e
	create V internal nodes p_i
	for all i in 1 to V do
		let e_i = selection on e with d = v_i
		buildTree(e_i, p_i)
```

### Training Set Error

Is the error of classification of the items in the test set. It's not zero due to the limit of the decision tree in general, some test on attribute values can fail.

This is a lower limit of the error we can expect when classifying new data

### Test Set Error

This is more indicative of the expected behavior with new data, and could be much worse than the Training Set Error

## Overfitting

Overfitting happens when the learning is affected by noise, so that the performance on the test set is (much) worse than that on the training set.

A decision tree is an hypothesis of the relationship between the predictor attributes and the class

- $h$ = hypothesis
- $error_{train}(h)$ = error of the hypothesis on the training set
- $error_{\epsilon}(h)$ = error of the hypothesis on the entire dataset

$h$ overfit the training set if there is an alternative hypothesis $h'$ such that:

$$error_{train}(h) < error_{train}(h') \\
error_{\epsilon}(h) > error_{\epsilon}(h')$$

The causes of overfitting are

- Presence of noise: individuals in the test set can have bad values in the predicting attributes and/or in the class label
- Lack of representative instances: some situation of the real world can be underrepresented in the training set

***Everything should be made as simple as possible, but not simpler***: all other things being equal, simple theories are preferable to complex ones

## Pruning a Decision Tree

Is a way to simplify it

- **Pre-pruning:** early stop of the tree growth, before it perfectly classify the training set
- **Post-pruning:** build a complete tree, then prune portion of it according to some criterion

In general post-pruning is preferred

### Validation set

The supervised data are split it three ***independent*** parts:

- Training set: the basis for building the classification model
- Validation set: the model is tuned (e.g. pruned) to minimize the error
- Test set: to assess the final expected error

### Statistical Pruning

- Error estimation: does the pruning reduce the maximum error expected?
- Significance testing: is the contribution of a node compatible with a random effect?

### Minimum Description Length - MDL

- The learning process produce a theory $T$  on a set of data, that can be encoded
    - $L(T)$ = length of the encoding of $T$ (in bit)
- The errors can also be encoded as exceptions to the theory
    - $L(\epsilon|T)$ = length of the encoding of the exceptions (in bit)
- The length of the theory is given by the sum of the encoding of the theory and the encoding of the exceptions

According to the MDL principle, the theory with the shortest description is to be preferred.

We are looking for the *most probable theory*, that maximize $P(T|\epsilon) = \frac{P(\epsilon|T) P(T)}{P(\epsilon)}$, or minimize its negative logarithm. With given $P(\epsilon)$ it becomes the minimum of $-log(P(\epsilon|T)) - log(P(T))$

# Decision Tree Learning Algorithm Options

## 1. Specification of the test for an attribute

It depends on the value domain of the attribute

### Discrete Domain

if the domain is discrete with *V* values, the split generates *V* nodes

### Continuous Domain

If the domain if continuous with *V* values, the split into *V* nodes is infeasible because the high number of branches would generate very small subsets and the significance would decrease very rapidly. Solutions:

- Discretization of the domain: each original value is mapped to the most appropriate discrete value, in general according to the distribution of the attribute values in the dataset elements
- Binarization: extreme case of discretization where the number of discrete values is 2 ⇒ threshold

### Threshold-based split

Let $d \in D$ be a real-valued attribute, let $t$ be a value of the domain of $d$, computed by sorting the records, counting the number of element below and above, let $c$ be the class attribute.

We define the entropy of $c$ w.r.t. $d$ with threshold $t$ as:

$$H(c|d:t) = H(c|d < t)*P(d<t) + H(c|d \geq t)*P(d \geq t)$$

We define

$$IG(c|d:t) = H(c) - H(c|d:t)$$

the Information Gain provided when we know if, for an individual, $d$ exceeds the threshold $t$ in order to forecast the class value

We define

$$IG(c|d) = max_tIG(c|d:t)$$

The complexity of computing this is $N *logN + 2*N*V_d$, where $N$ is the number of individuals in the node under consideration and $V_d$ is the number of distinct values of the attribute $d$. That's because of the sorting and the iteration in computing the $IG$ for each value of $t$.

## 2. Choice of the attribute to split the dataset

Looking for the split generating the maximum *purity.*

Measure of the purity:

- a node with two classes in the same proportion has low purity
- a node with only one class has highest purity

Some impurity functions are:

- Information Gain
- Gini Index
- Misclassification error

### Gini Index

Consider a node $p$ with $C_p$ classes

It measures the total probability of wrong classification given by random assignment based only on the class frequencies, $f_{p, j}$:

$$GINI_p = 1 - \sum_{j \in C_p} f_{p, j}^2$$

The maximum value is when all the records are uniformly distributed over all the classes: $1 - 1/C_p$

The minimum value is when all the records belong to the same class: $0$

### Misclassification error

Is the complement to 1 of the accuracy of a leaf, that is the highest frequency of the label indicated as output of the leaf

### Comparison

The behavior of ME is linear, therefore an error in the frequency is completely transferred into the impurity computation

Entropy and Gini have varying derivative, with the minimum around the center, so they are more robust w.r.t. errors in the frequency, when the frequencies of the two classes are similar

# Complexity of a Decision Tree

$N$ instances and $D$ attributes in $\epsilon$ → The tree height is $O(logN)$

Each level of the tree requires the consideration of all the dataset, and each node requires the consideration of all the attributes

The overall cost is $O(DNlogN)$

In addition the binary split of numeric attributes and pruning are $O(NlogN)$ that dosn't increase complexity

# Characteristics of DT

- It's a **non-parametric** approach to build classification models
- Finding the best DT is NP-complete
- The run-time use of a DT to classify new instances is extremely efficient: $O(h)$
- Robust w.r.t noise if appropriately pruned
- Redundant attributes do not cause any difficulty, because if two attributes have strong correlation, if one is chosen for a split, most likely the other will never provide a good increment of node purity, and will never be chosen
- The pruning strategy has high impact on the final result
- Portion of the tree are easily replicated, due to the **one attribute at a time** strategy
- They are prone to **overfitting**