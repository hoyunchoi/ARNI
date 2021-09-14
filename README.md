# ARNI: Algorithm for Reavealing Network Interactions

The code is originally forked from official repository for ARNI
https://github.com/networkinference/ARNI/tree/master/ARNI_Python


## What is ARNI
ARNI is algorithm introduced in "Jose C., Mor N., Sarah H., Marc T. Model-free inference of direct network interactions from nonlinear colleictive dynamics. Nat.Commun. (2017)"

For a given multi-variative time series data, it reavls the interaction between variables $i$ and $j$.
This algorithm only focuses on whether the interaction between two variables exists(active) or not(inactive).

When multi-variative time series data and target node is given to the algorithm, it returns the confidence value of other nodes having active link with the target node.

This confidence value can be evalutaed using ROC curve and AUC-score. The example code uses sklearn package for this evaluation method.

To make the example reproduction possible, simulation code for generating synthetic time series data is also included.

## Structure of Code
The core module of ARNI is only Basis.py and Reconstruction.py. If you already have some time series data to analysis and it's true connectivity pattern for evaluation, you may not use other modules.


### Common.py
This module contains Enum classes of available options. For detailed explanation, refer each modules

* NetworkList: Available topology of synthetic network.
    - ER
    - SF

* ModelList: Available models for synthetic simulation.
    - Kuramoto1
    - Kuramoto2
    - Michaelis-Menten
    - Roessler

* BasisList: Available basis for expanding pairwise interaction. Depending on target dynamics, appropriate basis exists.
    - Polynomial
    - PolynomialDiff
    - Fourier
    - FourierDiff
    - Power
    - RBF

### Network.py
This module generates synthetic network for simulation. The resulting weighted adjacency matrix has no self-loop and returned as numpy ndarray type.
Use networkx package for generation.

* ER: Erdos-Reyni random network, representing homogeneous topology
    - Directed: Use networkx.gnm_random_graph(directed=True)
    - Undirected: Use networkx.gnm_random_graph(directed=False)

* SF: Scale free network, representing heterogeneous topology
    - Directed: Use networkx.scale_free_graph
    - Undirected: Use network.barabasi_albert_graph

### Model.py
This model generates synthetic dynamics data under given weighted adjacency matrix $J_{ij}$.
Use scipy package for integration.

* Kuramoto1: Plain kuramoto model
    <img src="https://latex.codecogs.com/gif.latex? \dot{x}_i = \omega_i + \sum_j J_{ij} \sin(x_j-x_i)" />
* Kuramoto2: Kuramoto model with phase coupling
    <img src="https://latex.codecogs.com/gif.latex? \dot{x}_i = \omega_i + \sum_j J_{ij} \left[\sin(x_j-x_i) - 1.05 + 0.33 \sin(2(x_j-x_i))\right]" />
* Michaelis_Menten: Michaelis-Menten kinetics used in gene regulation
    <img src="https://latex.codecogs.com/gif.latex? \dot{x}_i = -x_i + \sum_j J_{ij} \frac{x_j}{1+x_j}" />
* Ressler: Coupled Rösler oscillator in chaotic regime
    <img src="https://latex.codecogs.com/gif.latex? \dot{x}_i = -y_i-z_i + \sum_j J_{ij} sin(x_j)" />
    <img src="https://latex.codecogs.com/gif.latex? \dot{y}_i = x_i + 0.1y_i" />
    <img src="https://latex.codecogs.com/gif.latex? \dot{z}_i = 0.1 + z_i(x_i-18.0)" />

### Basis.py
This module contains list of basis for pairwise interaction. Expands the dynamics of target node $i$ with chosen basis with chosen maximum order.

* Polynomial: Plain polynomial basis
    <img src="https://latex.codecogs.com/gif.latex? h^i_{j,p}=x_j^p" />
* PolynomialDiff: Polynomial basis with argument of difference
    <img src="https://latex.codecogs.com/gif.latex? h^i_{j,p}=(x_j-x_i)^p" />
* Fourier: Plain fourier basis
    <img src="https://latex.codecogs.com/gif.latex? h^{i, (1)}_{j,p} = \sin(px_j), h^{i, (2)}_{j,p} = \cos(px_j)" />
* FourierDiff: Fourier basis with argument of difference
    <img src="https://latex.codecogs.com/gif.latex? h^{i, (1)}_{j,p} = \sin(p(x_j-x_i)),  h^{i, (2)}_{j,p} = \cos(p(x_j-x_i))" />
* Power: Plain power basis
    <img src="https://latex.codecogs.com/gif.latex? h^i_{j,p_1,p_2} = x_i^{p_1}x_j^{p_2}" />
* RBF: Radial Basis Function
    <img src="https://latex.codecogs.com/gif.latex? h^i_{j,p} = 1+\Vert (x_i,x_j) - (x_{i,p}, x_{j,p}) \Vert^2" />

### Reconstruction.py
This module returns the confidence level of neighbor of target node estimated from raw simulation data.

Return value cost is MSE error between approximated time series and real times series data.

### Evaluation.py
This module evaluates the confidence level of ARNI with true (binary) connectivity pattern.

You can plot ROC curve and AUC score from evaluate method.




