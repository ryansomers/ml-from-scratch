# Derivations

These are the results the code relies on, derived from first principles. I use
column-vector conventions: $X \in \mathbb{R}^{n \times d}$ has one sample per row,
$w \in \mathbb{R}^{d}$, and $\nabla$ means the gradient with respect to the parameters.

## 1. Least squares and the normal equation

`mlfs/linear_regression.py` minimises the ridge objective (ordinary least squares is
$\alpha = 0$):

$$
J(w) = \lVert Xw - y \rVert^2 + \alpha \lVert w \rVert^2
     = (Xw - y)^\top (Xw - y) + \alpha\, w^\top w .
$$

Expanding, $J(w) = w^\top X^\top X w - 2 y^\top X w + y^\top y + \alpha w^\top w$. Using
$\nabla_w (w^\top A w) = 2Aw$ for symmetric $A$ and $\nabla_w (b^\top w) = b$:

$$
\nabla_w J = 2 X^\top X w - 2 X^\top y + 2\alpha w .
$$

Setting it to zero gives the **normal equation**

$$
(X^\top X + \alpha I)\, w^\star = X^\top y .
$$

The Hessian $2(X^\top X + \alpha I)$ is positive semi-definite, so $J$ is convex and any
stationary point is a global minimum; for $\alpha > 0$ or full-column-rank $X$ it is
positive definite and $w^\star$ is unique. I solve the system with `np.linalg.solve`
rather than forming $(X^\top X)^{-1}$: it is cheaper and loses less precision.

**Why QR is more stable.** With the reduced QR factorisation $X = QR$
($Q^\top Q = I$, $R$ upper triangular), the normal equation for OLS becomes
$R^\top Q^\top Q R w = R^\top Q^\top y$, i.e. (for invertible $R$) $R w = Q^\top y$, which back substitution
solves directly. This never forms $X^\top X$, whose condition number is
$\kappa(X)^2$. Ridge fits the same mould because

$$
\lVert Xw - y \rVert^2 + \alpha \lVert w \rVert^2
= \left\lVert \begin{bmatrix} X \\ \sqrt{\alpha}\, I \end{bmatrix} w
- \begin{bmatrix} y \\ 0 \end{bmatrix} \right\rVert^2 .
$$

**Householder reflections.** To zero column $k$ below the diagonal, let $x$ be that
sub-column and $v = x + \operatorname{sign}(x_1) \lVert x \rVert e_1$, normalised. The
reflection $H = I - 2 v v^\top$ is symmetric and orthogonal and maps $x$ to
$-\operatorname{sign}(x_1)\lVert x\rVert e_1$. Choosing the sign to match $x_1$ avoids
subtracting nearly equal numbers. Applying $H_1, \dots, H_d$ in turn gives
$H_d \cdots H_1 X = R$, so $Q = H_1 \cdots H_d$.

**Intercept.** I don't penalise the bias. Centring $X$ and $y$ makes the optimal bias
zero on the centred data, so I solve for $w$ there and recover
$b = \bar{y} - \bar{x}^\top w$.

**Gradient descent step size.** On $J/(2n)$ the gradient is
$\tfrac{1}{n}\left(X^\top (Xw - y) + \alpha w\right)$, which is Lipschitz with constant
$L = (\sigma_{\max}(X)^2 + \alpha)/n$. A step of $1/L$ guarantees monotone descent for a
convex $L$-smooth function, so I estimate $\sigma_{\max}^2$ by power iteration on
$X^\top X$ and use that.

## 2. Logistic regression gradient with L2

With $p_i = \sigma(z_i)$, $z_i = x_i^\top w + b$, $\sigma(z) = 1/(1 + e^{-z})$ and
labels $y_i \in \{0, 1\}$, the mean negative log-likelihood plus an L2 penalty is

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big]
+ \frac{\lambda}{2} \lVert w \rVert^2 .
$$

Two facts about the sigmoid do all the work: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$,
and therefore $\frac{d}{dz}\log\sigma(z) = 1 - \sigma(z)$ and
$\frac{d}{dz}\log(1 - \sigma(z)) = -\sigma(z)$. So for one sample

$$
\frac{\partial}{\partial z_i}\Big[-y_i \log p_i - (1-y_i)\log(1-p_i)\Big]
= -y_i (1 - p_i) + (1 - y_i) p_i = p_i - y_i .
$$

By the chain rule with $\partial z_i / \partial w = x_i$:

$$
\nabla_w J = \frac{1}{n} X^\top (p - y) + \lambda w,
\qquad
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_i (p_i - y_i).
$$

**Newton / IRLS.** Differentiating once more,
$\nabla_w^2 J = \frac{1}{n} X^\top S X + \lambda I$ with
$S = \operatorname{diag}(p_i(1 - p_i))$. Newton's update
$\theta \leftarrow \theta - H^{-1} \nabla J$ is the same as solving a weighted least
squares problem with weights $S$ at every step, hence "iteratively reweighted least
squares". Since $p(1-p) \le 1/4$, the gradient is Lipschitz with
$L \le \sigma_{\max}(X)^2/(4n) + \lambda$, which sets the default GD step.

**Matching scikit-learn.** sklearn minimises $C \sum_i \mathrm{NLL}_i + \tfrac12\lVert w\rVert^2$.
Dividing by $Cn$ gives my objective with $\lambda = 1/(Cn)$.

## 3. Softmax cross-entropy gradient

For logits $z \in \mathbb{R}^K$, $p_k = e^{z_k} / \sum_j e^{z_j}$ and a one-hot target
$y$, the loss is $\ell = -\sum_k y_k \log p_k$. Since
$\log p_k = z_k - \log \sum_j e^{z_j}$,

$$
\frac{\partial \log p_k}{\partial z_m} = \mathbb{1}[k = m] - p_m ,
$$

so

$$
\frac{\partial \ell}{\partial z_m} = -\sum_k y_k \big(\mathbb{1}[k=m] - p_m\big)
= -y_m + p_m \sum_k y_k = p_m - y_m .
$$

In matrix form for a batch, $\nabla_Z \bar{\ell} = \frac{1}{n}(P - Y)$. That is what the
fused `cross_entropy` op returns. With $Z = XW + \mathbf{1} b^\top$,
$\nabla_W = X^\top \nabla_Z$ and $\nabla_b = \mathbf{1}^\top \nabla_Z$, which is the
update `SoftmaxRegression` uses.

**Numerical stability.** $\log\sum_j e^{z_j} = c + \log\sum_j e^{z_j - c}$ for any $c$;
taking $c = \max_j z_j$ keeps every exponent $\le 0$, so nothing overflows.

**Log-softmax backward on its own.** If an upstream gradient $g = \partial L / \partial
(\log p)$ arrives, then
$\partial L / \partial z_m = \sum_k g_k (\mathbb{1}[k=m] - p_m) = g_m - p_m \sum_k g_k$,
which is the `log_softmax` backward rule in `mlfs/autograd/tensor.py`.

## 4. Backpropagation through an MLP layer

One layer computes, for a batch $X \in \mathbb{R}^{n \times d_{\text{in}}}$,

$$
Z = X W + \mathbf{1} b^\top, \qquad A = \operatorname{relu}(Z),
$$

with $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$. Suppose backprop has
already produced $G_A = \partial L / \partial A$ (same shape as $A$). Then:

**Through the ReLU** (elementwise, derivative $\mathbb{1}[Z > 0]$):

$$
G_Z = G_A \odot \mathbb{1}[Z > 0] .
$$

**Through the affine map.** Element-wise, $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$, so
$\partial Z_{ij}/\partial W_{kj} = X_{ik}$ and

$$
\frac{\partial L}{\partial W_{kj}} = \sum_i X_{ik} (G_Z)_{ij}
\;\;\Longrightarrow\;\;
G_W = X^\top G_Z .
$$

Similarly $\partial Z_{ij}/\partial X_{ik} = W_{kj}$ gives $G_X = G_Z W^\top$, which is
handed to the previous layer, and $b$ is broadcast over the $n$ rows, so its gradient
sums over them:

$$
G_b = \sum_{i=1}^{n} (G_Z)_{i,:} = G_Z^\top \mathbf{1} .
$$

The last line is the general rule the `Tensor` engine implements once in `unbroadcast`:
the adjoint of copying a value along an axis is summing along that axis. Matmul's rules
($G_U = G_C V^\top$, $G_V = U^\top G_C$ for $C = UV$) are the same two lines above.

A full MLP chains these: the output layer starts from $G_Z = (P - Y)/n$ (Section 3),
each layer produces $G_W, G_b$ for the optimiser and $G_X$ for the layer below.

## 5. The Adam update

For parameter $\theta$ with gradient $g_t$ at step $t$, Adam keeps exponential moving
averages of the gradient and its elementwise square, starting from
$m_0 = v_0 = 0$:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 .
$$

**Bias correction.** Unrolling, $m_t = (1 - \beta_1) \sum_{s=1}^{t} \beta_1^{t-s} g_s$.
If the gradients have a stationary mean $\mathbb{E}[g]$, then

$$
\mathbb{E}[m_t] = \mathbb{E}[g]\,(1 - \beta_1)\sum_{s=1}^{t}\beta_1^{t-s}
= \mathbb{E}[g]\,(1 - \beta_1^t),
$$

so $m_t$ underestimates the mean early on because of the zero start. Dividing fixes it:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} .
$$

**Update.**

$$
\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} .
$$

Each coordinate's step is the mean gradient divided by its root-mean-square, so the
step size is roughly $\eta$ regardless of the gradient's scale. On the first step
$\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the step is exactly
$\eta \operatorname{sign}(g_1)$ when $\epsilon = 0$; `tests/test_nn_optim.py` checks this.

For comparison, SGD with momentum (as I implemented it) is
$v_t = \mu v_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta v_t$: it averages
directions but does not rescale per coordinate.
