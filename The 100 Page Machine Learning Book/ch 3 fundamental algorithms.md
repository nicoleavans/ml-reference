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

At the time where the absence of computers required scientists to perform manual calculations, a linear classification model was desired. They discovered that if we define a negative label as
$0$
and the positive label as
$1$
, We would just need to find a  simple continuous function whose codomain is
$(0,1)$
. In such a case, if the value returned by the model for input 
$x$ 
is closer to 
$0$
, then we assign a negative label to
$x$
; otherwise, the example is labeled as positive. One function that has such a property is the **standard logistic function** (or **sigmoid function**):

$$
f(x) = \frac{1}{1+e^{-x}}
$$

where
$e$
is the base of the natural logarithm (also called *Euler's number*; 
$e^x$
is also known as the 
$exp(x)$
function in programming languages). It is depicted as:


<p align="center">
<img src="https://i.imgur.com/nSwT56v.png" width="600">
</p>

The logistic regression model looks like this:

$$
f_{\textbf{w},b}(\textrm{x}) \stackrel{\text{\tiny def}}{=} \frac{1}{1+e^{-(\textbf{w}\textrm{x}+b)}}
$$

The term
$w\textrm{x}+b$
is also from linear regression.

By looking at the graph of the standard logistic function, we can see how well it fits our classification purpose: if we optimize the values of
$w$
and
$b$
appropriately, we could interpret the output of 
$f(x)$
as the probability of
$y_i$
being positive. For example, if it's higher than or equal to the threshold
$0.5$
we would say that the class of
$\textrm{x}$
is positive; otherwise, it's negative. In practice, the choice of the threshold could be different depending on the problem. 

How do we find optimal
$w^*$
and 
$b^*$
? In linear regression, we minimized the empirical risk which was defined as the average squared error loss, also known as the **mean squared error** or MSE. 

### Solution
In logistic regression, we maximize the **likelihood** of our training set according to the model. In statistics, the likelihood function defines how likely the observation (an example) is according to our model.

For instance, let's have a labeled example
$(\mathrm{x}_i, y_i)$
in our training data. Assume also that we found (guessed) some specific values
$\hat{\mathrm{\textbf{w}}}$
and 
$\hat{b}$
of our parameters. If we now apply our model 
$f_{\hat{\mathrm{\textbf{w}}} , \hat{b}}$
to 
$\mathrm{x}_i$
using the logistic regression model, we will get some value
$0 < p < 1$ 
as output. If 
$y_i$
is the positive class, the likelihood of 
$y_i$
being the positive class, according to our model, is given by
$p$
Similarly, if 
$y_i$
is the negative class, the likelihood of it being the negative class is given by
$1-p$
.

The optimization criterion in logicstic regression is called **maximum likelihood**. Instead of minimizing the average loss, like in linear regression, we now maximize the likelihood of the training data according to our model:

$$
L_{\mathrm{w}, b} \stackrel{\text{\tiny def}}{=} \prod_{i=1...N} f_{\mathrm{w},b}
(\mathrm{x}_i)^{y_i}(1 - f_{\mathrm{w},b}(\mathrm{x}_i))^{(1 - y_i)}
$$

The expression
$f_{\mathrm{w},b}(\mathrm{x}_i)^{y_i}(1 - f_{\mathrm{w},b}(\mathrm{x}_i))^{(1 - y_i)}$
looks scary but is just a fancy mathematical way of saying: 
$f_{\mathrm{w},b}(\mathrm{x})$
when 
$y_i = 1$
and
$(1-f_{\mathrm{w},b}(\mathrm{x}))$
otherwise. Indeed, if 
$y_i = 1$
, then
$(1 - f_{\mathrm{w},b}(\mathrm{x}))^{(1 - y_i)}$ 
equals
$1$
because
$(1-y_i) = 0$
and we know that anything power 
$0$
equals
$1$
. On the other hand, if 
$y_i = 0$
then 
$f_{\mathrm{w},b}(\mathrm{x}_i)^{y_i}$
equals 
$1$
for the same reason.

You may have noticed we used the product operator
$\prod$
in the objective function instead of the sum operator
$\sum$
which was used in linear regression. It's because the lielihood of observing
$N$
labels for 
$N$
examples is the product of likelihoods of each observation (assuming that all observations are independent of one another, which is the case). You can draw a parallel with the multiplication of probabilities of outcomes in a series of independent experiments in the probability theory. 

Because of the
$exp$
function used in the model, in practice, to avoid **numerical overflow**, it's more convenient to maximize the **log-likelihood** instead of likelihood. The log-likelihood is defined as:

$$
LogL_{\mathrm{w},b} \stackrel{\text{\tiny def}}{=} \ln{(L_{\mathrm{w},b}(\mathrm{x}))}
= \sum^N_{i=1} [y_i \ln{f_{\textrm{w},b}(\mathrm{x})} + (1-y_i)\ln{(1-f_{\textrm{w},b}(\mathrm{x})})]
$$

Because 
$\ln$
is a **strictly increasing function**, maximizing this function is the same as maximizing its argument, and the solution to this new optimization problem is the same as the solution to the original problem. 

Contrary to linear regression, there's no closed form solution to the above optimization problem. A typical numerical optimization procedure used in such cases is **gradient descent**.

## Decision Tree Learning
A **decision tree** is an acyclic graph that can be used to make decisions. In each branching node of the graph, a specific feature
$j$
of the feature vector is examined. If the value of the feature is below a specific threshold, then the left branch is followed; otherwise, the right branch is followed. As the leaf node is reached, the decision is made about the class to which the example belongs. 

A decision tree can be learned from data.

### Problem Statement
Like previously, we have a collection of labeled examples; labels belong to the set
$\{0,1\}$
. We want to build a decision tree that would allow us to predict the class given a feature vector.

### Solution

There are various forumulations of the decision tree learning algorithm. Here, we consider one called **ID3**. 

The optimization criterion, in this case, is the average log-likelihood:

$$
\sum^N_{i=1} [y_i \ln{f_{ID3}(\mathrm{x}_i)} + (1-y_i)\ln{(1-f_{ID3}(\mathrm{x}_i)})]
$$

where
$f_{ID3}$
is a decision tree.

By now, it looks very similar to logistic regression. However, contrary to the logistic regression learning algorithm which builds a **parametric model** 
$f_{\mathrm{w}^*,b^*}$
by finding an *optimal solution* to the optimization criterion, the ID3 aglorithm optimizes it *approximately* by constructing a **nonparametric model**
$f_{ID3}(\mathrm{x}) \stackrel{\text{\tiny def}}{=} Pr(y= 1|\mathrm{x})$

The ID3 learning algorithm works as follows. Let
$\mathcal{S}$ 
denote a set of labeled examples. In the beginning, the decision tree only has a start node that contains all examples: 
$\mathcal{S} \stackrel{\text{\tiny def}}{=} \{(\mathrm{x}_i,y_i)\}^N_{i = 1}$
. Start with a constant model
$f^S_{ID3}$
defined as:

$$
f^S_{ID3} \stackrel{\text{\tiny def}}{=} \frac{1}{|\mathcal{S}|} \sum_{(\mathrm{x},y)\in \mathcal{S}} y
$$

The prediction given by the above model would be the same for any input
$\mathrm{x}$
. The corresponding decision tree is built using a toy dataset of 
$12$
labeled examples
$^4$
:

<p align="center">
<img src="https://i.imgur.com/EHCC1mz.png" width="250">
</p>

Then we search through all features
$j = 1,..., D$
and all thresholds
$t$
, and split the set 
$S$
into two subsets:
$\mathcal{S}_- \stackrel{\text{\tiny def}}{=} \{(\mathrm{x}, y)|(\mathrm{x}, y)\in\mathcal{S},x^{(j)}<t\}$
and
$\mathcal{S}_+ \stackrel{\text{\tiny def}}{=} \{(\mathrm{x}, y)|(\mathrm{x}, y)\in\mathcal{S},x^{(j)}\geq t\}$
. The two new subsets would go to two new leaf nodes, and we evaluate, for all possible pairs
$(j,t)$
, how good the split with pieces
$\mathcal{S}_-$
and
$\mathcal{S}_+$
is. 

Finally, we pick the best such values
$(j,t)$
, split 
$\mathcal{S}$
into 
$\mathcal{S}_+$
and
$\mathcal{S}_-$
, form two new leaf nodes, and continue recursively on 
$\mathcal{S}_+$
and
$\mathcal{S}_-$
(or quit if no split produces a model that's sufficiently better than the current one). A decision tree after one split is illustrated
$^5$
:

<p align="center">
<img src="https://i.imgur.com/GdPN01B.png" width="550">
</p>

Now you should wonder what "evaluate how good the split is" means. In ID3, the goodness of a split is estimated by using the criterion called **entropy**. Entropy is a measure of uncertainty about a random variable. It reaches its maximum when all values of the random variables are equiprobable. Entropy reaches its minimum when the random variable can only have one value. The entropy of a set of examples
$\mathcal{S}$
is given by:

$$
H(\mathcal{S}) \stackrel{\text{\tiny def}}{=} -f^{\mathcal{S}}_{ID3} \ln{f^{\mathcal{S}}_{ID3}} - (1- f^{\mathcal{S}}_{ID3})\ln{(1- f^{\mathcal{S}}_{ID3})}
$$

When we split a set of examples by a certain feature
$j$
and a threshold
$t$
, the entropy of a split
$H(\mathcal{S}_-, \mathcal{S}_+)$
, is simply a weighted sum of two entropies:

$$
H(\mathcal{S}_-, \mathcal{S}_+) \stackrel{\text{\tiny def}}{=} \frac{|\mathcal{S}_-|}{|\mathcal{S}|}H(\mathcal{S}_-) + \frac{|\mathcal{S}_+|}{|\mathcal{S}|}H(\mathcal{S}_+)
$$

So, in ID3, at each step, at each leaf node, we find a split that minimizes the entropy given by the above equation or we stop at this leaf node.

The algorithm stops at a leaf node in any of the following situations:

* All examples in the leaf node are classified correctly by the one-piece model (constant model $f^S_{ID3}$)
* We cannot find an attribute to split upon.
* The split reduces entropy less than some $\epsilon$ (the value for which has to be found experimentally).
* The tree reaches some maximum depth $d$ (also has to be found experimentally).

Because in ID3, the decision to split the dataset on each iteration isn't local (doesn't depend on future splits), the algorithm doesn't guarantee an optimal solution. The model can be improved using techniques like *backtracking* during the search for the optimal decision tree at the cost of possibly taking longer to build a model.

The most widely used formulation of a decision tree learning algorithm is called **C4.5**. It has several additional features as comared to ID3:
* it accepts both continuous and discrete features;
* it handles incomplete examples;
* it solves overfitting problems by using a bottom-up technique known as "pruning". 

Pruning consists of going back through the tree once it's been created and removing branches that don't contribute significantly enough to the error reduction by replacing them with leaf nodes. 

The entropy-based split criterion intuitively makes good sense: entropy reaches its minimum of 
$0$
when all examples in 
$\mathcal{S}$
have the same label; on the other hand, the entropy is at its maximum of
$1$
when exactly one-half of examples in 
$\mathcal{S}$
is labeled with 
$1$
, making such a leaf useless for classification. The only remaining question is how this algorithm approximately maximizes the average log-likelihood criterion, left for further reading.

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

$^4$
In the beginning, the decision tree only contains the start node; it makes the same prediction for any input.

$^5$
The decision tree after the first split; it tests whether feature
$3$
is less than 
$18.3$
and, depending on the result, the prediction is made in one of the two leaf nodes.

# Sources
* [The Hundred-Page Machine Learning Book](https://themlbook.com/) by Andriy Burkov, 2019 
* https://math.mit.edu/~dspivak/files/symbols-all.pdf
* https://detexify.kirelabs.org/classify.html
* https://www.steveklosterman.com/over-under/
