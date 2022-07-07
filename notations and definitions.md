# Notations and Definitions

### Vectors, Matrices, and Sets

| term | definition | denoted |
| --- | --- | ---- |
| scalar | a simple numerical value like an int or double | an italic letter like $x$ or $a$ |
| vector | an ordered list of scalar values, called attributes. vectors can be visualized as arrows that point to some direction, as well as points in a multi-dimensional space | *vector* - bold character like $\textbf{x}$ or $\textbf{a}$ *vector attribute* - italic value with an index like $w^{(j)}$ or $x^{(j)}$. index $j$ denotes a specific **dimension** of the vector, the position of an attribute in the list.* |
| matrix | a rectangular array of numbers arranged in rows and columns | bold capital letters, such as $\textbf{A}$ or $\textbf{W}$ | 
| set | an unordered collection of unique elements. a set of numbers can be finite (represented as {1,3,18,23,235} or {$x_1$,$x_2$,$x_3$}) or infinite and include all values in some interval (represented as [$a$,$b$] for all values between a and b. if the set doesn't include the values a and b it is shown as ($a$,$b$))** | a calligraphic capital character like $\mathcal{S}$ |

When an element $x$ belongs to a set $\mathcal{S}$, we write $x\in\mathcal{S}$. We can obtain a new set $\mathcal{S}_3$ as the **intersection** of two sets $\mathcal{S}_1$ and $\mathcal{S}_2$. In this case, we write $\mathcal{S}_3\leftarrow\mathcal{S}_1\cap\mathcal{S}_2$. For example, $\{1,3,5,8\}\cap\{1,8,4\}$ gives the new set $\{1,8\}$. An intersection consists of all numbers that are in both sets. 

We can obtain a new set $\mathcal{S}_3$ as a  **union** of two sets $\mathcal{S}_1$ and $\mathcal{S}_2$. For example, $\{1,3,5,8\}\cup\{1,8,4\}$ gives the new set $\{1,3,4,5,8\}$. A union includes all numbers in sets $\mathcal{S}_1$ or $\mathcal{S}_2$.

### Capital Sigma Notation

The summation over a collection $\mathcal{X}=\{x_1,x_2,...,x_n\}$ is denoted:

$$
\sum^{n}_{i=1}x_i\stackrel{\text{\tiny def}}{=}x_i+x_2+...+x_n
$$

or over the attributes over a vector $\textbf{x}=[x^{(1)},x^{(2)},...,x^{(m)}]$ is denoted:
$$
\sum^{m}_{j=1}x^{(j)}\stackrel{\text{\tiny def}}{=}x^{(1)}+x^{(2)}+...+x^{(m)}
$$

### Capital Pi Notation
Analagous to capital sigma is the capital pi notation. It denotes a product of elements in a collection or attributes of a vector:
$$
\prod^n_{i=1}x_i\stackrel{\text{\tiny def}}{=}x_1\cdot x_2\cdot ... \cdot x_n
$$
where $a\cdot b$ means a multiplied by b, frequently denoted simply as ab.

### Operations on Sets
A derived set creation operator looks like:
$$
\mathcal{S}^\prime\leftarrow\{x^2 | x \in\mathcal{S},x>3\}
$$

This means we create a new set $\mathcal{S}^\prime$ by putting into it $x^2$ such that $x$ is in $\mathcal{S}$, and $x$ is greater than 3.

The cardinality operator $|\mathcal{S}|$ returns the number of elements in set $\mathcal{S}$.

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
It may be denoted as $\textbf{w}\cdot\textbf{x}$. The two vectors must be of the same dimensionality, otherwise, the dot-product is undefined.

#### Vector - Matrix Multiplication

The multiplication of a matrix $\textbf{W}$ by a vector $\textbf{x}$ results in another vector. As an example, let our matrix be:
$$
\textbf{W} = \begin{bmatrix}w^{(1,1)} & w^{(1,2)} & w^{(1,3)} \\ w^{(2,1)} & w^{(2,2)} & w^{(2,3)}\end{bmatrix}
$$
When vectors participate in operations on matrices, a vector is by default represented as a matrix with one column. When the vector is on the right of the matrix, it remains a column vector. We can only multiply a matrix by vector if the vector has the same number of rows as the number of columns in the matrix. Let our vector be $\textbf{x}\stackrel{\text{\tiny def}}{=}[x^1,x^2,x^3]$. Then, $\textbf{Wx}$ is a two-dimensional vector defined as:
$$
\textbf{Wx}=\begin{bmatrix} w^{(1,1)} & w^{(1,2)} & w^{(1,3)} \\ w^{(2,1)} & w^{(2,2)} & w^{(2,3)}
\end{bmatrix} \begin{bmatrix} x^{(1)} \\ x^{(2)} \\x^{(3)} \end{bmatrix}\stackrel{\text{\tiny def}}{=}\begin{bmatrix} w^{(1,1)}x^{(1)} & w^{(1,2)}x^{(2)} & w^{(1,3)}x^{(3)} \\ w^{(2,1)}x^{(1)} & w^{(2,2)}x^{(2)} & w^{(2,3)}x^{(3)}
\end{bmatrix} = \begin{bmatrix}\textbf{w}^{(1)}\textbf{x} \\\textbf{w}^{(2)}\textbf{x}\end{bmatrix}
$$

If our matrix had y rows, the result of the product would be a y-dimensional vector.

When the vector is on the left side of the matrix in the multiplication, then it has to be **transposed** before we multiply it to the matrix. The transpose of vector $\textbf{x}$ denoted as $\textbf{x}^\intercal$ makes a row vector out of a column vector. For example:
$$
\textbf{x}=\begin{bmatrix}
x^{(1)} \\ x^{(2)}
\end{bmatrix},\quad \text{then} \quad \textbf{x}^\intercal=\begin{bmatrix} x^{(1)} & x^{(2)}
\end{bmatrix}
$$
The multiplication of the vector $\textbf{x}$ by the matrix $\textbf{W}$ is given by $\textbf{x}^\intercal\textbf{W}$.
$$
\textbf{x}^\intercal\textbf{W}=\begin{bmatrix} x^{(1)} & x^{(2)}
\end{bmatrix}\begin{bmatrix}
w^{(1,1)} & w^{(1,2)} & w^{(1,3)} \\ w^{(2,1)} & w^{(2,2)} & w^{(2,3)}
\end{bmatrix}
\newline\stackrel{\text{\tiny def}}{=}\begin{bmatrix}
w^{(1,1)}x^{(1)} + w^{(2,1)}x^{(2)},w^{(1,2)}x^{(1)}+w^{(2,2)}x^{(2)},w^{(1,3)}x^{(1)}+w^{(2,3)}x^{(2)}
\end{bmatrix}
$$
We can only multiply a vector by a matrix if the vector has the same number of dimensions as the number of rows in the matrix.

### Functions
A **function** is a relation that associates each element $x$ of a set $\mathcal{X}$, the **domain** of the function, to a single element $y$ of another set $\mathcal{Y}$, the **codomain** of  a function. A function usually has a name. If the function is called $f$, this relation is denoted $y=f(x)$ The element $x$ is the argument or input of the function, and $y$ is the value of the function on the output. The symbol that is used for representing the input is the variable of the function.

## Footnotes
\* Note: A variable can have two or more indices such as $x_i^{(j)}$ or $x_{i,j}^{(k)}$. In neural networks, we denote $x_{l,u}^{(j)}$ the input feature $j$ of unit $u$ in layer $l$.

\*\* A special set denoted $\mathbb{R}$ contains all numbers from minus infinity to plus infinity.

# Sources
* [The Hundred-Page Machine Learning Book](https://themlbook.com/) by Andriy Burkov, 2019 
* https://www.probabilitycourse.com/chapter1/1_2_2_set_operations.php
* https://math.mit.edu/~dspivak/files/symbols-all.pdf