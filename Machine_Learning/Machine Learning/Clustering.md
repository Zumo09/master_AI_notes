# Clustering

# The problem of clustering

***Given*** a set of *N* objects, each described by *D* values (the dimension), ***find*** a *natural* partitioning in *K* clusters and, possibly, a number of noise objects. The ***result*** is a *clustering scheme*, i.e. a function mapping each data to the sequence [1 . . . *K*] (or to noise). The objects in the same cluster should be similar (maximize the intra-cluster similarity), whereas objects in different clusters should be different (minimize the inter-cluster similarity)

# K-means

- Ask the user the number of cluster *K*
- Random choice of *K* points as **temporary centers**
- Each point finds his nearest center, and labels according to it
- For each center find the centroid of its points and move there the center
- Stop when the centroids do not move

We are trying to minimize the distortion

$$Distortion = \sum_{i=1}^N(e_i - Decode(Encode(e_i)))^2$$

To get the minimal distortion:

- $e_i$ must be encoded with the nearest center
- each center must be the centroid of the points it owns

### Algorithm termination

There is only a finite number of ways to partition *N* objects into *K* groups, and is finite also the number of configuration where all the centers are the centroids of the points they own.

If after one iteration the state changes, the distortion is **reduced** (convex function), therefore each change of state brings to a state which was never visited before. 

In summary, sooner or later the algorithm will stop because there are no new states reachable.

But the ending state isn't necessarily the best possible.

The starting point is important. The randomly chosen starting points should be as far as possible form each other. It may help to re-run the algorithm with different starting point.

Also the number of clusters *K* is an hyperparameter that must be handled carefully, trying various values and using a *quantitative evaluation* of the quality of the clustering scheme.

### The proximity function

The most obvious solution is the ***Euclidean distance***

### Sum of Squared Errors

The official name of distortion:

$$SSE = \sum_{j=1}^KSSE_j = \sum_{j=1}^K\sum_{i \in OwnedBy(\bold{c_j})}(e_i - \bold{c_j})^2$$

A cluster *j* with high $SSE_j$ has low quality

$SSE_j = 0 \iff$ all the points are coincident with the centroid

$SSE$ decreases for increasing *K*, is 0 when *K=N*, so minimizing *SSE* is not a viable solution to choose the best *K.*

It may happen that a centroid does not own any point. So a new centroid must be chosen:

- choose a point far away from the empty centroid
- choose a random point in the cluster with the highest SSE, so as the lower quality cluster will be split in two

### Outliers

Are points with high distance from their centroid, so they give high contribution to SSE

Sometimes is a good idea to remove them, because this algorithm has no notion of error, and assign a label to each object, but the choice is related to the application domain

### Complexity

*T* number of iterations, *K* number of clusters, *N* number of data points, *D* number of dimensions:

The time complexity is $O(TKND)$

## Pro and Cons

- Pro: Efficient: nearly linear in the number of data points (*T, K, D << N*)
- Cons: Cannot work with nominal data, where there is no concept of distance
- Requires the K parameter
- Sensitive to outliers
- does not deal with noise
- Does not deal properly with non convex clusters

# Evaluation of a clustering scheme

It is related only to the results, not to the technique. Clustering is a **non supervised** method, so we need indexes to measure various properties

## Measurement criteria

- Cohesion - proximity of objects in the same cluster should be high
- Separation between two cluster - several choices:
    - distance between the **nearest** objects in the two clusters
    - distance between the **most distant** objects in the two clusters
    - distance between the **centroids** of the two clusters

## Cohesion - prototype based

The sum of the proximity between the elements of the cluster and the geometric center (the prototypes), that could be:

- centroid - a point in the space whose coordinates are the mean of those of the dataset
- medoid - an element of the dataset whose average dissimilarity is minimal

$$Coh(k_i) = \sum_{x \in k_i} Prox(x, \bold{c}_i)$$

## Separation - prototype based

Separation between two cluster: proximity between the prototypes.

$$Sep(k_i, k_j) = Prox(\bold{c}_i, \bold{c}_j)$$

## Global separation of a clustering scheme

Sum of Squares Between clusters

$$\bold{c} = global.centroid.of.the.dataset \\
SSB = \sum_{i=1}^K N_iDist(\bold{c}_i,\bold{c})^2$$

- TTS = Total Sum of Squares : sum of squared distances of the points from the global centroid
- TSS = SSE + SSB

The total sum of squares is a global property of the dataset, independent from the clustering scheme.

## Silhouette index of a cluster

For the *i-th* object, compute the average distance w.r.t. the other objects of the same cluster; this value is $a_i$

For the *i-th* object and for each cluster other than it's own, compute the average distance w.r.t. the objects of the cluster; find the minimun w.r.t all the clusters; this value is $b_i$

For the *i-th* object the silhouette index is 

$$s_i = \frac{b_i - a_i}{max(a_i, b_i)} \in [-1, 1]$$

For the global index of a cluster/clustering scheme compute the average index over the cluster/dataset.

**Intuition:** when the index is less than 0 for an object it means that there is a dominance of objects in other clusters at a distance smaller than the objects in the same cluster.

### Looking for the best number of clusters

SSE and Silhouette are influenced by the number of clusters, so they can be used to optimize *K.*

Computation of silhouette is expensive. SSE decreases monotonically for increasing *K*

- *K = 1* ⇒ SSE = TSS
- *K = N* ⇒ SSE = 0

Plotting the SSE and the Silhouette for varying *K*: we find the best value of *K* where there is a slope change in SSE and a maximum in the Silhouette.

## Supervised measures

Let be available a partition $P = \{P_1, ..., P_L\}$ which we call **gold standard**, similar to the labelled data for training a classifier.

Considering a clustering scheme $K = \{k_1, ..., k_K\}$, we want to compare it to the gold standard in order to validate the clustering technique which can be applied later to new, unlabeled data.

### Classification oriented measures

Measure how the classes are distributed among the clusters

- Confusion Matrix
- precision, recall, f-measure

### Similarity oriented measures

Any pair of objects can be labelled as:

- SS if they belong to the same set in *P* and *K* : # of such pairs = $a$
- SD if they belong to the same set in *K* but not in *P* : # of such pairs = $b$
- DS if they belong to the same set in *P* but not in *K*: # of such pairs = c
- DD if they belong to different sets both in *P* and *K* : # of such pairs = $d$

**Rand Index** $R = \frac{a+d}{a+b+c+d}$

**Jaccard Coefficient** $J = \frac{a}{a+b+c}$

# Hierarchical Clustering

Generate a **nested structure** of clusters.

- Agglomerative (bottom up)
    - as a starting state, each data point is a cluster
    - in each step the two **less separated** clusters are merged into one
    - a measure of **separation between clusters** is needed
- Divisive (top down)
    - as a starting state, the entire dataset is the only cluster
    - in each step, the cluster with the lowest cohesion is split
    - a measure of cluster cohesion and split procedure are needed

The output is a so called *Dendrogram*.

The separation between clusters can be computed in several ways:

Graph based

- Single Link: $Sep(k_i, k_j) = \min_{x \in k_i, y \in k_j} Dist(x, y)$
- Complete Link: $Sep(k_i, k_j) = \max_{x \in k_i, y \in k_j} Dist(x, y)$
- Average Link: $Sep(k_i, k_j) = \frac1{|k_i||k_j|}\sum_{x \in k_i, y \in k_j} Dist(x, y)$

Prototype based

- Distance between the centroids
- Ward's method: difference between the total SSE resulting in case of merging the two clusters, and the sum of the original SSE. If the increase is small, then the separation is small.

## Single linkage hierarchical clustering

- Initialize the clusters, one for each objects
- Compute the **distance matrix** between the clusters. It's a squared, symmetric, N x N matrix, with the main diagonal null
- while the number of clusters is greater than 1
    - find the two clusters with lowest separation, say $k_r, k_S$
    - merge them into a cluster
    - delete from the distance matrix the rows and the columns *r* and *s* and insert a new row and column with the distances of the new cluster from the others

    $$Dist(k_k, k_{r+s}) = \min(Dist(k_k, k_r), Dist(k_k, k_s)) \forall{k} \in [1, K] $$

Space and time complexity:

- $O(N^2)$ for the computation of the distance matrix
- worst case *N - 1* iterations to reach the final cluster
- for the *i-th* step:
    - search of the pair to merge $O((N-i)^2)$
    - computation of the new row of the dist. matrix $O(N-i)$

In summary $O(N^3)$, which can be reduced, with indexing structure, to $O(N^2log(N))$

### Generating the clustering scheme

The desired clustering scheme is obtained by **cutting** the dendrogram at some level of the **total dissimilarity** inside the clusters, which increase for decreasing number of clusters.

Single linkage tend to generate cluster with a larger diameter, while Complete linkage generates more compact clusters.

## Summary

🙁 The scaling is poor, due to the high complexity

🙁 There isn't a global objective function, the decision is always local and cannot be undone

🙂 The dendrogram structure is of great help for the interpretation of the result

🙂 Empirically, the result is frequently good

# Density Based Clustering

Clusters are high-density regions separated by low-density regions

Computing the density:

- Grid-based:
    - split the (hyper)space into a regularly spaced grid
    - count the number of objects inside each grid element
- Object-centered:
    - define the radius of a (hyper)sphere
    - attach to each object the number of objects which are inside that sphere

## DBSCAN - Density Based Spatial Clustering of Application with Noise

- Define a radius $\epsilon$ as define as neighborhood of a point the $\epsilon$-hypersphere centered at that point.
    - Neighborhood is symmetric: if *p* is neighbor of *q*, then also *q* is in the neighborhood of *p*.
- Define a threshold $*minPoints*$ and define as **core** a point with at least $*minPoints$* points in its neighborhood, as **border** otherwise
- Define that a point *p* is **directly density reachable** from point *q* if
    - *q* is core
    - *q* is in the neighborhood of *p*
    - Direct density reachability is not symmetric
- A point *p* is **density reachable** from point *q* iff
    - *q* is core
    - there is a sequence of points $q_i$ such that $q_{i+1}$ is directly density reachable from $q_i, i \in [1, nq], q_1$ is directly reachable from $*q*$ and $*p*$ is directly density reachable form $q_{nq}$.
    - Reachability is not symmetric
- A point $p$ is **density connected** to point $q$ iff there is a point $s$ such that $p$ and $q$ are density reachable from $s$
    - density connection is symmetric

A *cluster* is a maximal set of point connected by density

Border points which are not connected by density to any core point are labelled as **noise**

### Algorithm

```python
Require: SetOfPoints: UNCLASSIFIED points
Require: Eps, MinPts
	ClusterId <- nextId(NOISE);
	for i = 0 to SetOfPoints.size do
		Point <- SetOfPoints.get(i)
		if Point.ClId = UNCLASSIFIED then
			if ExpandCluster(SetOfPoint, Point, ClusterId, Eps, MinPts)
			then
				ClusterId <- next(ClusterId)
Ensure: SetOfPoints

ExpandCluster(SetOfPoint, Point, ClId, Eps, MinPts) : Boolean;
	seeds := SetOfPoints.regionQuery(Point, Eps);
	IF seeds.size < MinPts THEN # no core point
		SetOfPoint.changeClId(Point, NOISE);
		RETURN False;
	ELSE
		# all points in seed are density-reachable from Point
		SetOfPoints.changeClId(seeds, ClId);
		seeds.delete(Point);
		WHILE seeds <> Empty DO
			currentP := seeds.first();
			result := SetOfPoints.regionQuery(currentP, Eps);
			IF result.size >= MinPts THEN
				FOR i FROM 0 TO result.size DO
					resultP := results.get(i);
					IF resultP.ClId IN {UNCLASSIFIED, NOISE} THEN
						IF resultP.ClId = UNCLASSIFIED THEN 
							seeds.append(resultP);
						END IF;
						SetOfPoints.changeClId(resultP, ClId);
					END IF; # Unclassified or noise
				END FOR;
			END IF; # result.size >= MinPts
			seed.delete(currentP);
		END WHILE; # seeds <> Empty
		RETURN True;
	END IF;
END;
```

### Comment

🙂 Find clusters of any shape

🙂 Is robust w.r.t. noise

🙁 Problem if clusters have widely varying densities

Being based on distances between points, the complexity is $\mathcal{O}(N^2)$, reduced to $\mathcal{O}(Nlog(N))$ if spatial indexes are available

Very sensitive to the values of $\epsilon$ and $minPoints$.

Decreasing $\epsilon$ and increasing $minPoints$ reduces the cluster size and increases the number of noise points.

## Kernel Density Estimation

Describe the distribution of the data by a function. The overall density function is the sim of the **influence functions** (or the **kernel functions**) associated with each point

The kernel function must be **symmetric** and monotonically decreasing, usually has a *parameter* to set the decreasing rate.

### DENCLUE algorithm

- Derive a density function for the space  occupied by the data points
- Identify the points that are local maxima
- Associate each point with a density attractor by moving in the direction of maximum increase in density
- Define clusters consisting of points associated with a particular density attractor
- Discard clusters whose density attractor has a density less than a user-specified threshold $\xi$
- Combine clusters that are connected by a path of points that all have a density of $\xi$ or higher

🙂 It has a strong theoretical foundation on statistics

DBSCAN is a particular case of DENCLUE where the influence is a step function

🙂 Good at dealing with noise and clusters of different shapes and sizes

🙁 Expensive computation $O(N^2)$

🙁 Troubles with high dimensional data and clusters with different densities

# Model Based Clustering

Estimate the parameters of a statistical model to maximize the ability of the model to **explain the data**

The base model is usually the multivariate normal, and the estimation is usually done using the maximum likelihood

## Expectation Maximization

A general case where the data can have many mixed distributions

- Select an initial set of model parameters
- Repeat
    - Expectation Step - for each object, calculate the probability that each object belongs to each distribution
    - Maximization Step - Given the probabilities from the expectation step, find the new estimates of the parameter that maximize the expected likelihood
- Until - the parameter do not change, or the change is below a specified threshold.

### Example one dimension, two clusters

Ned to estimate 5 parameters

- mean and standard deviation for cluster A
- mean and standard deviation for cluster B
- sampling probability *p* for cluster A

$$Pr(A|x) = \frac{Pr(x|A)Pr(A)}{Pr(x)} = \frac{f(x; \mu_A, \sigma_A)p_A}{Pr(x)} \\
f(x; \mu_A, \sigma_A) = \frac1{\sqrt{2\pi}\sigma}e^{\frac{(x - \mu)^2}{2\sigma^2}}$$

Compute the numerator for $Pr(A|x)$ and $Pr(B|x)$ and normalize dividing by their sum

Assign $x$ to the cluster with maximum probability

Expectation: compute $p_A$

Maximization of the distribution likelihood given the data:
Compute distributions parameters, each data point has a *weight* which is the probability of belonging to that distribution

 

$$\mu_A = \frac{w_1e_1 + ... + w_Ne_N}{w_1 + ... + w_N} \\
\sigma_A = \frac{w_1(e_1 - \mu_A)^2+ ... + w_N(e_N - \mu_A)^2}{w_1 + ... + w_N}$$