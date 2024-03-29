# Notations and Definitions

### Vectors, Matrices, and Sets

| term | definition | denoted |
| --- | --- | ---- |
| scalar | a simple numerical value like an int or double | an italic letter like $x$ or $a$ |
| vector | an ordered list of scalar values, called attributes. vectors can be visualized as arrows that point to some direction, as well as points in a multi-dimensional space | *vector* - bold character like $\textbf{x}$ or $\textbf{a}$ *vector attribute* - italic value with an index like $w^{(j)}$ or $x^{(j)}$. index $j$ denotes a specific **dimension** of the vector, the position of an attribute in the list.* |
| matrix | a rectangular array of numbers arranged in rows and columns | bold capital letters, such as $\textbf{A}$ | 
| set | an unordered collection of unique elements. a set of numbers can be finite (represented as {1,3,18,23,235} or {$x_1$,$x_2$,$x_3$}) or infinite and include all values in some interval (represented as [$a$,$b$] for all values between a and b. if the set doesn't include the values a and b it is shown as ($a$,$b$))** | a calligraphic capital character like $\mathcal{S}$ |

When an element 
$x$
belongs to a set 
$\mathcal{S}$
, we write 
$x\in\mathcal{S}$
. We can obtain a new set 
$\mathcal{S}_3$
as the **intersection** of two sets
$\mathcal{S}_1$
and 
$\mathcal{S}_2$
. In this case, we write 
$\mathcal{S}_3\leftarrow\mathcal{S}_1\cap\mathcal{S}_2$
. For example, 
$\{1,3,5,8\}\cap\{1,8,4\}$
gives the new set 
$\{1,8\}$
. An intersection consists of all numbers that are in both sets. 

We can obtain a new set 
$\mathcal{S}_3$
as a  **union** of two sets 
$\mathcal{S}_1$
and 
$\mathcal{S}_2$
. For example, 
$\{1,3,5,8\}\cup\{1,8,4\}$
gives the new set 
$\{1,3,4,5,8\}$
. A union includes all numbers in sets 
$\mathcal{S}_1$
or 
$\mathcal{S}_2$.

### Capital Sigma Notation

The summation over a collection $\mathcal{X}=\{x_1,x_2,...,x_n\}$ is denoted:

$$
\sum^{n}_{i=1}x_i\stackrel{\text{\tiny def}}{=}x_i+x_2+...+x_n
$$

or over the attributes over a vector 
$\textbf{x}=[x^{(1)},x^{(2)},...,x^{(m)}]$
is denoted:

$$
\sum^{m}_{j=1}x^{(j)}\stackrel{\text{\tiny def}}{=}x^{(1)}+x^{(2)}+...+x^{(m)}
$$

### Capital Pi Notation
Analagous to capital sigma is the capital pi notation. It denotes a product of elements in a collection or attributes of a vector:

$$
\prod^n_{i=1}x_i\stackrel{\text{\tiny def}}{=}x_1\cdot x_2\cdot ... \cdot x_n
$$

where 
$a\cdot b$
means a multiplied by b, frequently denoted simply as ab.

### Operations on Sets
A derived set creation operator looks like:

$$
\mathcal{S}^\prime\leftarrow\{x^2 | x \in\mathcal{S},x>3\}
$$

This means we create a new set $\mathcal{S}^\prime$ by putting into it $x^2$ such that $x$ is in $\mathcal{S}$, and $x$ is greater than 3.

The cardinality operator 
$|\mathcal{S}|$
returns the number of elements in set $\mathcal{S}$.

### Operations on Vectors
The sum of two vectors $\textbf{x} + \textbf{z}$ is defined as the vector:

$$
[x^{(1)}+z^{(1)},x^{(2)}+z^{(2)},...,x^{(m)}+z^{(m)}]
$$

The difference of two vectors $\textbf{x} - \textbf{z}$ is defined as:

$$
[x^{(1)}-z^{(1)},x^{(2)}-z^{(2)},...,x^{(m)}-z^{(m)}]
$$

A vector multiplied by a scalar is a vector. Example:

$$
\textbf{x}c\stackrel{\text{\tiny def}}{=}[cx^{(1)},cx^{(2)},...,cx^{(m)}]
$$

A **dot-product** of two vectors is a scalar. Example:

$$
\textbf{wx}\stackrel{\text{\tiny def}}{=}\Sigma^m_{i=1}w^{(i)}x^{(i)}
$$

It may be denoted as 
$\textbf{w}\cdot\textbf{x}$
. The two vectors must be of the same dimensionality, otherwise, the dot-product is undefined.

#### Vector - Matrix Multiplication

The multiplication of a matrix 
$\textbf{W}$
by a vector 
$\textbf{x}$
results in another vector. As an example, let our matrix be:

$$
\bf{W} = \begin{bmatrix}w^{(1,1)} & w^{(1,2)} & w^{(1,3)} \\\\ w^{(2,1)} & w^{(2,2)} & w^{(2,3)}\end{bmatrix}
$$

When vectors participate in operations on matrices, a vector is by default represented as a matrix with one column. When the vector is on the right of the matrix, it remains a column vector. We can only multiply a matrix by vector if the vector has the same number of rows as the number of columns in the matrix. Let our vector be 
$\textbf{x}\stackrel{\text{\tiny def}}{=}[x^1,x^2,x^3]$
. Then, 
$\textbf{Wx}$
is a two-dimensional vector defined as:

$$
\textbf{Wx}=\begin{bmatrix} w^{(1,1)} & w^{(1,2)} & w^{(1,3)} \\\\ w^{(2,1)} & w^{(2,2)} & w^{(2,3)}
\end{bmatrix} \begin{bmatrix} x^{(1)} \\\\ x^{(2)} \\\\ x^{(3)} \end{bmatrix}
\stackrel{\text{\tiny def}}{=}\begin{bmatrix} w^{(1,1)}x^{(1)} & w^{(1,2)}x^{(2)} & w^{(1,3)}x^{(3)} \\\\ w^{(2,1)}x^{(1)} & w^{(2,2)}x^{(2)} & w^{(2,3)}x^{(3)}
\end{bmatrix}
$$

$$
= \begin{bmatrix} \textbf{w}^{(1)}\textbf{x} \\\\ \textbf{w}^{(2)}\textbf{x} \end{bmatrix}
$$

If our matrix had y rows, the result of the product would be a y-dimensional vector.

When the vector is on the left side of the matrix in the multiplication, then it has to be **transposed** before we multiply it to the matrix. The transpose of vector 
$\textbf{x}$
denoted as 
$\textbf{x}^\intercal$
makes a row vector out of a column vector. For example:

$\textbf{x}$
$=\begin{bmatrix}
x^{(1)} \\ x^{(2)}
\end{bmatrix},\quad \text{then} \quad \textbf{x}^\intercal=\begin{bmatrix} x^{(1)} & x^{(2)}
\end{bmatrix}$

The multiplication of the vector 
$\textbf{x}$
by the matrix 
$\textbf{W}$
is given by 
$\textbf{x}^\intercal\textbf{W}$
.

$$
\textbf{x}^\intercal\textbf{W}=\begin{bmatrix} x^{(1)} & x^{(2)}
\end{bmatrix}\begin{bmatrix}
w^{(1,1)} & w^{(1,2)} & w^{(1,3)} \\\\ w^{(2,1)} & w^{(2,2)} & w^{(2,3)}
\end{bmatrix}
\newline\stackrel{\text{\tiny def}}{=}\begin{bmatrix}
w^{(1,1)}x^{(1)} + w^{(2,1)}x^{(2)},w^{(1,2)}x^{(1)}+w^{(2,2)}x^{(2)},w^{(1,3)}x^{(1)}+w^{(2,3)}x^{(2)}
\end{bmatrix}
$$

We can only multiply a vector by a matrix if the vector has the same number of dimensions as the number of rows in the matrix.

### Functions
A **function** is a relation that associates each element 
$x$
of a set 
$\mathcal{X}$
, the **domain** of the function, to a single element $y$ of another set 
$\mathcal{Y}$
, the **codomain** of  a function. A function usually has a name. If the function is called 
$f$
, this relation is denoted 
$y=f(x)$
The element $x$ is the argument or input of the function, and 
$y$
is the value of the function on the output. The symbol that is used for representing the input is the variable of the function. (Thus, x is the variable of the function 
$f$
. 

We say that 
$f(x)$
has a **local minimum** at
$x=c$
if 
$f(x) \geq f(c)$
for every 
$x$
in some open interval around
$x=c$
.
|     |     |
| --- | --- |
| interval | a set of real numbers with the property that any number that lies between two numbers in the set is also included in the set |
| open interval | does not include its endpoint and is denoted using parentheses. ie $(0,1)$ means 'all numbers greater than $0$ and less than $1$|
| global minimum | the minimal value among all the local minima |

A vector function, dented as
$\bf{y} = \bf{f}(x)$ 
is a function that returns a vector
$\bf{y}$ 
. It can have a vector or scalar argument.

### Max and Arg Max
Given of set of values
$\mathcal{A} = \{a_1,a_2, ..., a_n\}$
, the operator 
$\max_{a\in\mathcal{A}}f(a)$
returns the highest value
$f(a)$
for all elements in the set
$\mathcal{A}$
. Whereas the operator 
${\mathrm{argmax}}(a)_{a\in\mathcal{A}}f(a)$
returns the element of the set 
$\mathcal{A}$
that maximizes
$f(a)$
.

Sometimes, when the set is implicit or infinite, we can write
$\max_af(a)$
or 
${\mathrm{argmax}}(a)$
.

Operators 
$\min$
and 
${\mathrm{argmin}}$
function similarly.

### Assignment Operator
The expression
$a\leftarrow f(x)$
means that the variable
$a$
gets the new value: the result of 
$f(x)$
. We say that the variable 
$a$
gets assigned a new value. Similarly, 
$\bf{a}$
$\leftarrow [a_1,a_2]$
means that the vector variable
$\bf{a}$
gets the two-dismensional vector value
$[a_1,a_2]$
.

### Derivative and Gradient
A **derivative** 
$f^\prime$
of a function
$f$
is a function or a value that describes how fast 
$f$
grows (or decreases). If the derivative is a constant value, like
$5$
or
$-3$
, then the function grows (or decreases) constantly at any point
$x$
of its domain. If the derivative
$f^\prime$
is a function, then the function
$f$
can grow at a different pace in different regions of its domain. If the derivative
$f^\prime$
is positive at some point
$x$
, then the function 
$f$
grows at this point. If the derivative of
$f$
is negative at some
$x$
, then the function decreases at this point. The derivative of 
$0$
at 
$x$
means that the function's slope at 
$x$
is horizontal.

The process of finding a derivative is called **differentiation**.

Derivatives for basic functions are known. For example, if
$f(x) = x^2$
, then
$f^\prime(x) = 2x$
; if 
$f(x) = 2x$
, then
$f^\prime(x) = 2$
; if
$f(x) = 2$
, then
$f^\prime(x) = 0$
(the derivative of any function
$f(x) = c$
, where
$c$
is a constant value, is zero).

If the function we want to differentiate is not basic, we can find its derivative using the **chain rule**. (It is also possible to find partial derivatives with the chain rule.) For instance, if
$F(x) = f(g(x))$
, where
$f$
and
$g$
are some functions, then 
$F^\prime (x)= f^\prime (g(x)g^\prime (x))$
. For example, if 
$F(x) = (5x+1)^2$
, then 
$g(x) = 5x+1$
and
$f(g(x)) = (g(x))^2$
. By applying the chain rule, we find:

$$F^\prime (x) = 2(5x+1)g^\prime (x) = 2(5x +1)5 = 50x+10$$

**Gradient** is the generalization of derivative for functions that take several inputs (or one input in the form of a vector or some other complex structure). A gradient of a function  is a vector of **partial derivatives**. You can look at finding a partial derivative of a function as the processs of finding the derivative by focusing on one of the function's inputs and by considering all other inputs as constant values.

For example, if our function is defined as 
$f([x^{(1)},x^{(2)}]) = ax^{(1)} + bx^{(2)} + c$
, then the partial derivative of function
$f$
*with respect to*
$x^{(1)}$
, denoted as
$\frac{\partial f}{\partial x^{(1)}}$
is given by,

$$
\frac{\partial f}{\partial x^{(1)}} = a + 0 + 0 = a
$$

, where 
$a$
is the derivative of the function
$ax^{(1)}$
; the two zeroes are respectively derivatives of 
$bx^{(2)}$
and
$c$
, because
$x^{(2)}$
is considered constant when we compute the derivative with respect to 
$x^{(1)}$
, and the derivative of any constant is zero.

Similarly, the partial derivative of function
$f$
with respect to 
$x^{(2)}$
, 
$\frac{\partial f}{\partial x^{(2)}}$
,
is given by,

$\frac{\partial f}{\partial x^{(2)}} = 0 + b + 0 = b$

The gradient of function 
$f$
denoted as 
$\triangledown f$
is given by the vector

$$
[ \frac{\partial f}{\partial x^{(1)}},\frac{\partial f}{\partial x^{(2)}} ] 
$$

### Random Variable

a **random variable**, usually written as an italiic capital letter, like *X*, is a variable whose possible values are numerical outcomes of a random phenomenon. Examples include a coin toss, dice roll, or the height of the first stranger you see. There are two types of random variables, **discrete** and **continuous**.

| | | 
| -- | -- |
| discrete random variable | takes on only a countable number of distinct values such as $red, yellow, blue$ or $ 1, 2, 3 $ | 
| probability distribution | described by a list of probabilities associated with each of its possible values. this list of values is called **probability mass function** (pmf)*** |
| continuous random variable (crv) | takes an infinite number of possible values in some interval, such as height, weight, or time. because the number of values of a continus random variable $X$ is infinite, the probability of any option is $0$. the probability distribution of a crv is described by a **probability density function** (pdf). the pdf is a function whose codomain is nonnegative and the area under the curve is equal to $1$ | 

Let a discrete random variable *X* have
$k$
possible values 
$\{x_i\}^k_{i = 1}$
. The **expectation** of *X* denoted as 
$\mathbb{E}[X]$
is given by:

$$
\mathbb{E}[X] \stackrel{\text{\tiny def}}{=} \sum^k_{i=1} [x_i \cdot Pr(X=x_i)] 
\newline = x_1 \cdot Pr(X = x_2) + ... + x_k \cdot Pr(X = x_k)
$$

where
$Pr(X = x_i)$
is the probability that *X* has the value 
$x_i$
accoding to the pmf. The expectation of a random variable is also called the **mean**, **average**, or **expected value** and is frequently denoted by the letter
$\mu$
. The expectation is one of the most important **statistics** of a random vairable.

Another important statistic is the **standard deviation**, defined as:

$$
\sigma \stackrel{\text{\tiny def}}{=} \sqrt{\mathbb{E}[(X - \mu)^2]}
$$

**Variance**, denoted as
$\sigma^2$
or 
${\textrm{var}} (X)$
is defined as:

$$
\sigma^2 = \mathbb{E}[(X- \mu)^2]
$$

For a discrete random variable, the standard deviation is given by:

$$
\sigma = \sqrt{PR(X = x_1)(x_1 - \mu)^2 + PR(X = x_2)(x_2 - \mu)^2 + ... +Pr(X = x_k)(x_k - \mu)^2}
$$

where 
$\mu = \mathbb{E}[X]$
. The expectation of a continuous random variable *X* is given by:

$$
\mathbb{E}[X] \stackrel{\text{\tiny def}}{=} \int_\mathbb{R} xf_{\normalsize\mathrm{x}}(x)dx
$$

where 
$f_{\normalsize\mathrm{x}}$
is the pdf of the variable *X* and
$\int_\mathbb{R}$
is the integral of function
$xf_{\normalsize\mathrm{x}}$
. Integral is an equivalent of the summation over all values of the function when the function has a continuous domain. It equals the area under the curve of the function. The property of the pdf that the area under its curve is 
$1$
mathematically means that
$\sum_\mathbb{R} f_{\normalsize\mathrm{x}}(x)dx = 1 $
. Most of the time we don't know
$f_{\normalsize\mathrm{x}}$
, but we can observe some values of *X*. In machine learning, we call these values **examples**, and the collection of these examples is called a **sample** or a **dataset**.

### Unbiased Estimators
Because
$f(x)$
is usually unknown, but we have a sample
$S_\mathrm{x} = \{x_i\}^N_{i=1}$
, we ofen content ourselves not with the true values of statistics of the probability distribution, such as expectation, but with their unbiased estimators.

We say that:

| if $\hat{\theta}(S_\mathrm{X})$ has the property: | $\mathbb{E}[\hat{\theta}(S_\mathrm{X})] = \theta$ |
| --- | ---- |
| $\theta$ | statistic calculated using a sample |
| $S_\mathrm{X}$ | sample drawn from an unknown probability distribution |
| $\hat{\theta}$ | **sample statistic** obtained using a sample and not the real statistic that can be obtained only from knowing $f(x)$ ; the expectation is taken over all possible samples drawn |
| $\hat{\theta}(S_\mathrm{X})$ | unbiased estimator of some statistic |

Intuitively, this meansthat if you can have an unlimited number of such samples as 
$S_\mathrm{X}$
, and you compute some unbiased estimator, such as
$\hat{\mu}$
, using each sample, then the average of all these
$\hat{\mu}$
equals the real statistic
$\mu$
that you would get computed on
$\mathrm{X}$
.

It can be shown that an unbiased estimator of an unknown
$\mathbb{E}[\mathrm{X}]$
(given by the expectation of either a discrete random variable or continuous random variable) is given by the **sample mean**:
$\frac{1}{N} \sum^N_{i=1}x_i$

### Bayes' Rule
The conditional probability 
$Pr(X = x | Y= y)$
is the probability of the random variable
$X$
to have a specific value
$x$
given that another random variable 
$Y$
has a specific value of
$y$
. The Bayes' Rule, or Bayes' Theorem, stipulates that:

$$
Pr(X = x | Y= y) = \frac{Pr(X = x | Y= y) | Pr(X = x)}{Pr(Y=y)}
$$

### Parameter Estimation
Bayes' Rule comes in handy when we have a model of 
$X$
's distribution, and this model
$f_{\theta}$
is a function that has some parameters in the form of a vector
$\boldsymbol{\theta}$
. An example of such a function could be the Gaussian function that has two parameters,
$\mu$
and
$\sigma$
, and is defined as\*\*\*\*:

$$
f_{\theta}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

where 
$\boldsymbol{\theta} \stackrel{\text{\tiny def}}{=} [\mu , \sigma]$
and
$\pi$
is the constant 
$3.14159...$
.

This function has all the properties of a pdf. Therefore, we can use it as a model of an unknown distribution of 
$X$
. We can update the values of the parameters in the vector 
$\boldsymbol{\theta}$
from the data using Baye's Rule:

$$
Pr(\theta = \hat{\theta} | X = x) \leftarrow \frac{Pr(X = x | \theta = \hat{\theta}) | 
Pr(\theta = \hat{\theta})}{Pr(X=x)} = \frac{Pr(X = x | \theta = \hat{\theta})Pr(\theta = \hat{\theta})}{\sum_{\tilde{\theta}}Pr(X=x|\theta = \hat{\theta})Pr(\theta = \hat{\theta})}
$$

where
$Pr(X=x | \theta = \hat{\theta}) \stackrel{\text{\tiny def}}{=} f_{\hat{\theta}}$
.

If we have a sample
$\mathcal{S}$
of 
$\mathrm{X}$
and the set of possible values for 
$\boldsymbol{\theta}$
is finite, we can estimate
$Pr(\theta = \hat{\theta})$
by applying Bayes' rule iteratively, one example
$x\in \mathcal{S}$
at a time. The initial value
$Pr(\theta = \hat{\theta})$
can be guessed such that 
$\sum_{\hat{\theta}}Pr(\theta = \hat{\theta})=1$
. This guess of the probabilities for different
$\hat{\theta}$
is called the **prior**.

First, we compute
$Pr(\theta = \hat{\theta}|X=x_1)$
for all possible values
$\hat{\theta}$
. Then, before updating 
$Pr(\theta = \hat{\theta}|X=x)$
again, this time for
$x=x_2\in \mathcal{S}$
using the Bayes' Rule established above, we replace the prior
$Pr(\theta = \hat{\theta})$
by the new estimate
$Pr(\theta = \hat{\theta})\leftarrow \frac{1}{N}\sum_{x\in \mathcal{S}} Pr(\theta = \hat{\theta}|X=x)$
.

The best value of the parameters
$\boldsymbol{\theta}^*$
given one example is obtained using the principle of **maximum a posteriori** (MAP):

$$
\boldsymbol{\theta}^* = \underset{\theta}{\mathrm{argmax}} \prod^N_{i=1}Pr(\theta = \hat{\theta}|X = x_1)
$$

If the set of possible values for 
$\theta$
isn't finite, then we need to optimize the above equation directly using a numerical optimization routine, such as a gradient descent. Usually, we optimize the natural logarithm of the right-hand side expression because the logarithm of a product becomes the sum of logarithms and it's easier for the machine to work with a sum than a product. (Multiplication of many numbers can give either a very small or very large result. Then, **numerical overflow** becomes an issue when the machine cannot store the numbers in memory.)

### Parameters vs. Hyperparameters
A hyperparameter is a property of a learning algorithm, ususally having a numerical value. That value influences the way the algorithm works. Hyperparameters aren't learned by the algorithm itself from the data. They have to be set by the data analyst before running the algorithm. 

Parameters are variables that define the model learned by the learning algorithm. Parameters are directly modified by the learning algorithm based on the training data. The goal of learning is to find such values of parameters that make the model optimal.

### Classification vs. Regression
**Classification** is a problem of automatically assigning a **label** to an unlabeled example, like assigning emails the label spam.

In machine learning, the classification problem is solved by a **classification learning algorithm** that takes a collection of labeled examples as inputs and produces a model that can take an unlabeled example as input and either directly output a label or a value that can be used to deduce the label by the analyst, such as a probability. 

In a classification problem, a label is a member of  finite set of **classes**. If the size of the set of classes is two, it is binary classification (or binomial). Multiclass classification (or multinomial) deal with three or more classes. There's still one label per example, though.

While some learning algorithms naturally allow for multinomials, others are by nature binomial. 

**Regression** is a problem of predicting a real-valued label (often called a **target**) given an unlabeled example. Estimating house price valuation based on house features is a famous example.

The regression problem is solved by a **regression learning algorithm** that takes a collection of labeled examples as inputs and produces a model as input and output a target.

### Model-Based vs. Instance-Based Learning
Most supervised learning algorithms are model-based, such as SVM. Model-based aglorithms use the training data to create a model that has parameters learned from the training data. In SVM, the two parameters we saw were
$\bf{w}^*$
and
$b^*$
. After the model was built, the training model can be discarded.

Instance-based learning algorithms use the whole dataset as the model. One instance-based algorithm frequently used in practice is **k-Nearest Neighbors** (kNN). In classification, to predict a label for an input example the kNN algorithm looks at the close neighborhood of the input example in the space of feature vectors and outputs the label that is saw the most often in this close neighborhood.

### Shallow vs. Deep Learning
A **shallow learning** algorithm learns the parameters of the model directly from the features of the training examples. Most supervised learning algorithms are shallow. The exceptions are **neural network** learning algorithms, specifically those that build neural networks with more than one layer between input and output. Such neural networks are called **deep neural networks**. In deep neural network learning (or deep learning), most model parameters are learned not directly from the features of the training examples, but from the outputs of the preceding layers.

## Footnotes
\* Note: A variable can have two or more indices such as 
$x_i^{(j)}$
or 
$x_{i,j}^{(k)}$
. In neural networks, we denote 
$x_{l,u}^{(j)}$
the input feature 
$j$
of unit $u$ in layer 
$l$.

\*\* A special set denoted 
$\mathbb{R}$
contains all numbers from minus infinity to plus infinity.

\*\*\* For example:

$$
Pr(X = blue) = 0.25, Pr(X = red) = 0.3, Pr(x = yellow) = 0.45
$$

The sum of probabilities equals 
$1$.

\*\*\*\* Defines the pdf of one of the most frequently used in practice probability distributions, called **Gaussian distribution** or **normal distribution** and denoted as
$\mathcal{N}(\mu , \sigma^2)$
.

# Sources
* [The Hundred-Page Machine Learning Book](https://themlbook.com/) by Andriy Burkov, 2019 
* https://www.probabilitycourse.com/chapter1/1_2_2_set_operations.php
* https://math.mit.edu/~dspivak/files/symbols-all.pdf
* https://detexify.kirelabs.org/classify.html