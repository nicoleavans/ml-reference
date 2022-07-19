# Fundamental Algorithms

There are five algorithms that are not just the most known but also either very effective on their own or used as building blocks for the most effective learning algorithms.

## Linear Regression
Linear regression is a popular regression learning algorithm that learns a model which is a linear combination of features of the input example.

### Problem Statement
We have a collection of labeled examples
${(x_i, y_i)}^N_{i=1}$
:

| term | definition |
| -- | -- |
| $N$ | size of the collection | 
| $x_i$ | $D$-dimensional feature vector of example $i=1,...,N$ |
| $y_i$ | a real-valued$^1$ target and every feature $x_i^{(j)}, j = 1, ... , D$ is also a real number. | 

We want to build a model 
$f_{w,b}(\textrm{x})$
as a linear combination of features of example
$x$
:

$$
f_{w,b}(\textrm{x}) = \textbf{\textrm{wx}} + b
$$

| term | definition | 
| -- | -- | 
| $\textbf{\textrm{w}}$ | is a *D*-dimensional vector of parameters |
| $b$ | a real number |
| $f$ | model |
| $f_{w,b}$ | notates that the model is parameterized by two values, w and b |

We will use the model to predict the unknown 
$y$
for a given
$\mathrm{x}$
like this:
$y \leftarrow f_{\mathrm{w},b}(x)$
. Two models parameterized by two different pairs
$(\mathrm{w},b)$
will likely produce two different predictions when applied to the same example. We want to find the optimal values 
$(\mathrm{w}^*,b^*)$
that define the model that makes the most accurate predictions.

The form of the linear model is very similar to the form of the SVM model. The only difference is the missing sign operator. The two models are similar, however the hyperplane in the SVM plays the role of the decision boundary (used to separate two groups of examples from one another). Thus, it needs to be as far from each group as possible.

Alternatively, the hyperplane in linear regression is chosen to be as close to all training examples as possible.

<p align="center">
<img src="https://i.imgur.com/WrOP1Dm.jpeg" width="600">
</p>

You can see why the latter requirement is essential by looking at the graph above. We can use this line to predict the value of the target
$y_{new}$
for a new unlabeled input example 
$x_{new}$
. If our examples are D-dimensional feature vectors (for
$D>1$
), the only difference with the one-dimensional case is that the regression model is not a line but a plane or hyperplane.

If the line in the figure above was far from the points, the prediction
$y_{new}$
would have fewer chances to be correct.

### Solution
To satisfy the requirement that the hyperplane in linear regression is chosen to be as close to all training examples as possible, the optimization procedure we use to find the optimal values for
$\textbf{w}^*$
and 
$b^*$
tries to minimize the following expression:

$$
\frac{1}{N} \sum_{i=1...N} (f_{w,b}(x_i)-y_i)^2
$$

In mathematics, the expression we minimize or maximize is called an **objective function**, or, simply, an objective. The expression
$(f_{w,b}(x_i)-y_i)^2$
in the above objective is called the **loss function**. It's a measure of penalty for misclassification of example
$i$
. This particular choice of the loss function is called **squared error loss**. All model-based learning algorithms have a loss function, and what we do to find the best model is try to minimize the objective known as the **cost function**. In linear regression, the cost function is given by the average loss, also known as **empirical risk**. The average loss for a model is the average of all penalties obtained by applying the model to the training data. 

Why is the loss in linear regression a quadratic function? Why couldn't we get the absolute value of the difference between the true target
$y_i$
and the predicted value 
$f(x_i)$
and use that as a penalty? We could, moreover, we could use a cube instead of a square. 

Now you probably start realizing how many seemingly arbitrary decisions are made when we design a machine learning algorithm: we decided to use the linear combination of features to predict the target. However, we could use a square or some other polynomial to combine the values of features. We could also use some other loss function that makes sense: the absolute difference between 
$f(x_i)$
and 
$y_i$
makes sense, the cube of the difference too; the **binary loss**
$(1$
when 
$f(x_i)$
and 
$y_i$ 
are different and 
$0$
when they are the same) also makes sense, right?

If we made different decisions about the form of the model, the form of the loss function, and about the choice of the algorithm that minimizes the average loss to find the best values of parameters, we would end up inventing a new learning algorithm. The fact that it's different doesn't mean that it will work better in practice.

People invent new learning algorithms for one of the two main reasons:

1. The new algorithm solves a specific practical problem better than existing algorithms.
2. The new algorithm has better theoretical guarantees on the quality of the model it produces.

One practical justification of the choice of the linear form of the model is that it's simple. Why use a complex model when you can use a simple one? Another consideration is that linear models rarely overfit. **Overfitting** is the property of a model such that the model predicts very well labels of the examples used during training but frequently makes errors when applied to examples that weren't seen by the learning algorithm during training. 

<p align="center">
<img src="https://i.imgur.com/pWBV3Jq.png" width="600">
</p>

This is an example of overfitting in regression. The data used to build the regression line is the same as in the previous figure, but this time, it is the polynomial regression with a polynomial of degree 
$10$
. The regression line predicts almost perfectly the targets of almost all training examples, but will likely make significant errors on new data. 

But what about the squared loss, why did we decide that it should be squared? It is convenient.
$^2$
The absolute value is not convenient, because it doesn't have a continuous derivative, which makes the function not smooth. Functions that are not smooth create unnecessary difficuties when employing linear algebra to find closed form solutions to optimization problems. Closed form solutions to finding an optimum of a function are simple algebraic expressions and are often preferable to using complex numerical optimization methods, such as **gradient descent** (used, among other applications, to train neural networks).

Intuitively, squared penalties are also advantageous because they exaggerate the difference between the true target and the predicted one according to the value of this difference. We might also use the powers 
$3$
or 
$4$
, but their derivatives are more complicated to work with.

Finally, why do we care about the derivative of the average loss? If we can calculate the gradient of the objective function, we can then set this gradient to zero
$^3$
and find the solution to a system of equation that gives us the optimal values
$\textbf{w}^*$
and 
$b^*$
.


# Footnotes

$^1$ 
To say that 
$y_i$
is real-valued, we write
$y_i \in \mathbb{R}$
, where
$\mathbb{R}$
denotes the set of all real numbers. 

$^2$
In 1805, the French mathematician Adrien-Marie Legendre, who first published the sum of squares method for gauging the quality of the model, stated that squaring the error before summing is *convenient*.

$^3$
To find the minimum or the maximum of function, we set the gradient to zero because the value of the gradient at extrema of a function is always zero. In 2D, the gradient an an extremum is a horizontal line.

## Logistic Regression
Logistic regression is not a regression, but a classification learning algorithm. The name comes from statistics and is due to the fact that the mathematical formulation of logistic regression is similar to that of linear regression. It is explained here in the case of binary classification, however, it can naturally be extended to multiclass classification.

### Problem Statement
In logistic regrssion, we still want to model
$y_i$
as a linear function of
$x_i$ 
, however, with a binary
$y$
this is not straightforward. The linear combination of features such as
$wx_i+b$
is a function that spans from
$-\infty$
to 
$\infty$
, while 
$y_i$
has only two possible values.

# Sources
* [The Hundred-Page Machine Learning Book](https://themlbook.com/) by Andriy Burkov, 2019 
* https://math.mit.edu/~dspivak/files/symbols-all.pdf
* https://detexify.kirelabs.org/classify.html
* https://www.steveklosterman.com/over-under/
