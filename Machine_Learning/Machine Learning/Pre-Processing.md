# Pre-Processing

## Aggregation

Combining two or more attributes (or objects) into a single attribute (or object)

Purpose: data reduction, change of scale, more stable data

## Sampling

Used when obtaining or processing the entire data set could be impossible or too expensive.

The sample is representative if it has approximately the same property of interest as the original set of data. In this case using a sample will work almost as well as using the entire data set.

### Type of sampling

- **Simple random**: a random choice with given probability distribution
- **With replacement**: repetition of independent extraction. In a small population a small subset could be underestimated
- **Without replacement**: repetition of extractions, removing the extracted one from the population
- **Stratified**: used when the data set is split into subset with homogeneous characteristics. Split the data into several partition according to some criteria, then draw the random sample from each partition, to guarantee the representativity inside each subset

Statistics provides techniques to assess the optimal sample size, with sample significativity, with a tradeoff between data reduction and precision.

# Dimensionality reduction

To avoid the *curse of dimensionality*: when dim is very high the occupation of space becomes very high

Noise reduction, and reduce time, memory and complexity of mining algorithms

## Principal Component Analysis PCA

Find projections that captures most of the data variations

- Eigenvectors if the covariance matrix, that define the new space

The new dataset will have only the attributes witch captures most of the data variation

## Features subset selection

Local way to reduce dimensionality by removing:

- Redundant attributes: that duplicate most of info contained in other attr.
- Irrelevant attributes: that does not contains useful info for the analysis

### Methods

- **Brute force**: try all possible features subset as input to the algorithm
- **Embedded approach**: feat. sel. occurs as part of the data mining algorithm
- **Filter approach:** feat. are selected before the alg. is run
- **Wrapper approach:** brute force but without an exhaustive search

## Feature creation

New features can captures more efficiently data characteristics

- Feature extraction: e.g. pixel in a image → distance between objects
- Mapping to a new space: e.g. Fourier transform
- New features

## Discretization

Continuous ⇒ Discrete: using thresholds. Only one threshold **binarization**.

Discrete with many values ⇒ discrete with less values: guided by domain knowledge

Binarization of discrete values: Attribute d allowing V values ⇒ V binary attributes

## Attribute transformation

Map the entire set of values to a new set according to a function, to change their distribution

- $x^k , log(x), e^x, |x|$

Standardization $x \rightarrow \frac{x - \mu}{\sigma}$: Translation and shrinking without changing in the distribution

Normalization: the domains are mapped to standard ranges

$$x \rightarrow \frac{x - x_{min}}{x_{max} - x_{min}} (0, 1), x \rightarrow \frac{x - \frac{x_{max} + x_{min}}{2}}{\frac{x_{max} - x_{min}}{2}} (-1, 1)$$

# Similarity and Dissimilarity

- Similarity
    - numerical measure of how alike two data objects are
    - higher when obj are more alike
    - often range [0, 1]
- Dissimilarity
    - contrary as before
    - range [0, upper] where upper could change

**Attribute type**

**Dissimilarity**

**Similarity**

Nominal

$$d = \begin{cases} 
0 &\quad p = q \\
1 &\quad p \neq q
\end{cases}$$

$$s = \begin{cases} 
1 &\quad p = q \\
0 &\quad p \neq q
\end{cases}$$

Ordinal
values mapped to integer 0 to V - 1

$$d = \frac{|p - q|}{V - 1}$$

$$s = 1 - \frac{|p - q|}{V - 1}$$

Interval or ratio

$$d = |p - q|$$

$$s = \frac{1}{1 + d} \\ or \\
s = 1 - \frac{d - min(d)}{max(d) - min(d)}$$

## Distances

### Euclidean distance - $L_2$

$$dist = \sqrt{\sum_{d=1}^D(p_d-q_d)^2}$$

Where $D$ is the number of dimensions and $p_d$ and $q_d$ are, respectively, the d-th component of the data objects $p$ and $q$

## Minkowski distance - $L_r$

$$dist = (\sum_{d=1}^D(p_d-q_d)^r)^\frac{1}{r}$$

Standardization or normalization is necessary if scales differ. $r$ is a parameter which is chosen depending on the data set and the application

- $r = 1$ : city block, Manhattan, $L_1$ norm
    - works better than euclidean in very high dimensional spaces
- $r = \infty$ : Chebyshev, $L_\infty$ norm:
    - consider only the dimensions where the difference is maximum

        $$dist_\infty = lim_{r\rightarrow \infty}(\sum_{d=1}^D(p_d-q_d)^r)^\frac{1}{r} = max_d |p_d - q_d|$$

## Mahalanobis distance

Consider data distribution, described by the covariance matrix

$$\Sigma_{ij} = \frac{1}{N - 1} \sum_{k=1}^N(e_{ki} - \bar{e}_i)(e_{kj} - \bar{e}_j) \\ 
dist_m = \sqrt{(p-q)\Sigma^{-1}(p-q)^T}$$

The Covariance matrix represent the variation of pairs of random variables, summed over all the observations. The main diagonal contains the variances.

The values are positive if the two variables grow together

If the matrix is diagonal, the variables are not correlated

If the variables are standardized, the diagonal contains "ones"

## Properties of a distance

- Positive definiteness: $Dist(p, q) \geq 0,\forall{p, q}, Dist(p, q) = 0 \iff p = q$
- Symmetry: $Dist(p, q) = Dist(q, p)$
- Triangle inequality: $Dist(p, q) \leq Dist(p, r) + Dist(r, q), \forall{p, q, r}$

A distance function satisfying all the properties above is called ***metric***

## Properties of Similarity

- *Sim(p, q) = 1 only if p = q*
- *Sim(p, q) = Sim(q, p)*

### Similarity between binary vectors

Consider the counts below:

- $M_{00}$ the # of attr. where p is 0 and q is 0
- $M_{01}$ the # of attr. where p is 0 and q is 1
- $M_{10}$ the # of attr. where p is 1 and q is 0
- $M_{11}$ the # of attr. where p is 1 and q is 1

**Simple Matching Coefficient**

$$SMC = \frac{number\_of\_matches}{number\_of\_attributes} = \frac{M_{00} + M_{11}}{M_{00} +M_{01} + M_{10} +M_{11}}$$

**Jaccard Coefficient**

$$JC = \frac{number\_of\_11\_matches}{number\_of\_non\_00\_attributes} = \frac{M_{11}}{M_{01} + M_{10} +M_{11}}$$

**Cosine similarity**

It is the cosine of the angle between two vectors

$$cos(p, q) = \frac{p \cdot q}{\lVert p \rVert \lVert q\rVert}$$

**Tanimoto**

Variation of Jaccard for continuous or count attributes. It reduces to Jaccard for binary attributes.

$$T(p, q) = \frac{p \cdot q}{\lVert p \rVert^2 +\lVert q\rVert^2 - p \cdot q}$$

## Correlation of quantitative data

Measure of the linear relationship between a pair of attributes. Consider as vector the ordered list of the values over all the data records, then standardize those vectors and compute the dot product

$$\bold{p} = [p_1, ..., p_N] \Rightarrow standardize \Rightarrow \bold{p'} \\ 
\bold{q} = [q_1, ..., q_N] \Rightarrow standardize \Rightarrow \bold{q'} \\
corr(p, q) = \bold{p' \cdot q'}$$

- Independent Varibles ⇒ Correlation Zero
    - The inverse is not valid in general
- Correlation Zero ⇒ absence of linear relationship between the variables

## Correlation between nominal attributes

$$U(p, q) = 2 \frac{H(p) + H(q) - H(p, q)}{H(p) + H(q)}$$

Where $H(\cdot)$ is the entropy of a single attribute while $H(\cdot,\cdot)$ is the joint entropy, computed from the joint probability.

It is always between 0 and 1