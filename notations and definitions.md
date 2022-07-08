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

# Sources
* [The Hundred-Page Machine Learning Book](https://themlbook.com/) by Andriy Burkov, 2019 
* https://www.probabilitycourse.com/chapter1/1_2_2_set_operations.php
* https://math.mit.edu/~dspivak/files/symbols-all.pdf
* https://detexify.kirelabs.org/classify.html
