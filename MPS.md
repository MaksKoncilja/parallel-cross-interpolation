# Quantics: MPS

To learn basics of quantics we follow the sequence:

1D function, binary grid, quantics tensor, TT/MPS compression, evaluation and reconstruction, integration.

## One-dimensional representation of a simple function $f(x)$

### Function reidexation into a tensor

Consider a simple function $f(x)$ defined on the interval $x \in [x_{min}, x_{max}]$. Choose an integer $N$, that denotes the number of binary bits used to encode grid index, which gives $M = 2^{N}$ grid points. A midpoint grid with spacing $h = \frac{1}{M}(x_{max}- x_{min})$ gives grid points 

$$ x_{n} = x_{min} + (n+ \frac{1}{2})h , \quad n = 0, \dots 2^{N}-1. $$

At this points the sampled vector is defined as  $y_{n} = f(x_{n})$, equivalently:

$$y = [f(x_0), f(x_1), \dots, f(x_{2^N-1})]^{T} $$

Binary-encoding the grid index $n$:

$$n = \sigma_{1} + 2 \sigma_{2} + 2^{2} \sigma_{3} + \dots 2^{N-1} \sigma_{N} = \sum_{i=1}^{N} 2^{i-1} \sigma_{i}, \quad \sigma_{i} \in \{0, 1\}$$

The bit $\sigma_{1}$ is the least significant, changing it's value moves value $n$ between neighboring grid points, the bit $\sigma_{N}$ on the other hand is the most significant as it changes $n$ for $2^{N-1}$ grid points. 

Binary-indexed tensor is defined as $F_{\sigma_1,\ldots,\sigma_N}=f(x_n)$. This is a reinterpretation of a vector $y_{n}$ of lenght $2^{N}$  as an order $N$ tesor of shape $(2, 2, \dots 2)$. 

The reidexation is the first step of the quantic representation,followed by approximation of the binary idexed tensor by a tensor train to compress the information. As a full tensor grows exponentially with $N$, replacing it with a product of small tensors, so called tensor train representation scales roughly linearly with $N$. 

The idea behind tensor train is to approximate large tensor as a product of much smaller object 

$$
F_{\sigma_{1}, \dots,  \sigma_{N}} \approx G_{1}(\sigma_{1}) G_{2}(\sigma_{2})  \dots G_{N}(\sigma_{N}),
$$
here, each $G_{i}(\sigma_{i})$ is a small matrix, or matrix-like object, that depends on the binary index $\sigma_{i}$. The first object $G_{1}(\sigma_{1})$ is a row vector, the intermediate object are matrices and the last object $G_{N}(\sigma_{N})$ is a column vector. The product gives an approximation of the function value at the grid point. 

The first TT implementation we construct tensor train using $SVD$ decomposition, starting from the full tensor $F_{\sigma_{1}, \dots,  \sigma_{N}}$, extracting tensor train cores one by one by performing singular value decompositions. Becouse the method requires building the full tensor the is not scalable, but it is a usefull learning example.


The dimensions of the intermediate matrix indices are called tensor-train ranks, or bond dimensions. These ranks measure how much information is passed from one part of the chain to the next.

If the ranks are small, the tensor train is a compact representation. If the ranks become large, the compression becomes inefficient.

A first way to construct the tensor train is to use singular value decompositions. This method is called TT-SVD.

It is important to frame this correctly: TT-SVD is not a different representation. The final representation is still the tensor train

The SVD is only the local matrix factorization used to obtain the tensor-train cores.

For a matrix $A$, the singular value decomposition is $A = U D V^{T}$. Matrix $D$ is diagonal with singular values, it tells the amount of information each component of the matrix carries. Discarding the smallest gives a low-rank approximation. To build Matrix Product States ($MPS$) from the quantics tensor we repeatedly apply SVD decomposition. Firstly the full quatics tensor is reshaped into a matrix $F_{\sigma_{1}, (\sigma_{2}, \dots, \sigma_{N})}$ with size $2 \times 2^{N-1}$, secondly we apply $SVD$ decomposition  $U_{1} D_{1} V_{1}$ gaining the first tensor train core $G_{1}(1, \sigma_{1}, \alpha_{1})=U_{1}(\sigma_{1}, \alpha_{1})$. The number of kept singular values in the diagonal matrix $D$ gives the first bond dimension $r_{1}$, the first bond index we donete as $\alpha_{1}$. The remaining object $S_{1}V_{1}^{T}$ with shape $(\alpha_{1}, (\sigma_{2}, \dots, \sigma_{N}))$ is reshaped to $R_{1}(\alpha_{1}, \sigma_{2}, \dots, \sigma_{N})$ and finaly to $A_{2}((\alpha_{1}, \sigma_{2}), (\alpha_{3}, \dots,  \sigma_{N}))$ before SVD is appied $A_{2}=U_{2}D_{2}V_{2}^{T}$. Keeping $r_{2}$ singular values, the second TT core can be computed by reshaping $U((\alpha_{1}, \sigma_{2}), \alpha_{2})$ into $G_{2}(\alpha_{1}, \sigma_{2}, \alpha_{2})$. 

In general, step $k$ is started with $R_{k-1}(\alpha_{k-1}, \sigma_{k}, \dots, \sigma_{N})$. 

Reshaping it into a matrx $A_{k}((\alpha_{k-1}, \sigma_{k}), (\sigma_{k+1}, \dots \sigma_{N}))$ with the size $(r_{k-1} \cdot 2) \times 2^{N-k}$. 

Apply SVD to decopose $A_{k} = U_{k} D_{k} V_{k}^{T}$, keep $r_{k}$ singular values, reshape $U_{k}((\alpha_{k-1}, \sigma_{k}), \alpha_{k})$ into the $k$-th TT core $G_{k}(\alpha_{k-1}, \sigma_{k}, \alpha_{k})$.  The remaining factor $S_{k}V_{k}^{T}(\alpha_{k}, (\sigma_{k+1}, \dots, \sigma_{N}))$  is reshaped into $R_{k}(\alpha_{k}, \sigma_{k+1}, \dots, \sigma_{N})$. 

This is repeated for $k = 1, \dots, N-1$. 

The final TT core is equal to the remaining tensor after constructing the ones that come before $G_{N}(\alpha_{N-1}, \sigma_{N},1)=R_{N-1}(\alpha_{N-1}, \sigma_{N})$.

Final TT representation is 

$$
F_{\sigma_{1}, \dots, \sigma_{N}} \approx \sum_{\alpha_{1}, \dots, \alpha_{N-1}} G_{1}(1, \sigma_{1}, \alpha_{1}) G_{2}(\alpha_{1}, \sigma_{2}, \alpha_{2}) \dots G_{N}(\alpha_{N-1}, \sigma_{N}, 1).
$$


If all the singular valueas are kept at every step, then the above expression is not an approxiamtion, but is exact. At each step $k$ the rank $r_{k}$ is determined by the number of singular values kept. For compressed decomposition the value $r_{k}$ is choosen such that the discarded singular values are small. Singular values a the step $k$ are $d_{k,1} \geq d_{k,2} \geq \dots,$ the discarded squared error at this step is then

$$
\sum_{j > r_{k}} s_{k,j}^{2}.
$$

Usual approach is to choose such $r_{k}$ the discarded squered error is below a prediscribed tolerance $\epsilon$. 

### Pointwise MPS evaluation and full tensor reconstruction

Pointwise MPS evaluates one tensor entry without full tensor reconstruction $F_{\sigma_{1}, \dots, \sigma_{N}} \approx y_{n} = f(x_{n})$.  

Binary indices $\sigma_{i}$ are calculated from $n$ by extracting binary digits of $n$:

$$
\sigma_{i} = \left\lfloor \frac{n}{2^{i-1}} \right\rfloor mod \; 2.
$$

Full tensor reconstruction $\tilde{F}_{\sigma_{1}, \dots, \sigma_{N}}$ gives approximate function value that we can comapre to the original full tensor $||F - \tilde{F}|| / || F ||$. 




