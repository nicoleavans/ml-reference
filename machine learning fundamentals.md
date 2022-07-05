# Types of Machine Learning
## Supervised Learning
In supervised learning, the dataset is a collection of labeled examples:

$$\{(x_i,y_i)\}^N_{i=1} $$

Each element $x_i$ among $N$ is called a feature vector, a vector in which each dimension contains a value that describes the example somehow. That value is called a feature and is denoted as $x^{(j)}$ 
For instance, if each example x in our collection represents a person, then the first feature, $x^{(1)}$ 
could contain height, the second feature $x^{(1)}$
could contain weight, etc. For all examples, the feature at position j always contains the same kind of information. 

The label $y_i$ can be either an element belonging to a finite set of classes, or a real number, or a more complex structure (vector, matrix, tree, graph, etc). You can see a class as a category to which an example belongs. For instance, if your examples are emails and your problem is spam detection, then you have two classes $\{spam, not\_spam\}$.

The goal of a supervised learning algorithm is to use the dataset to produce a model that takes a feature vector $x$ as input and outputs information deducing the label for this feature vector.
### How it Works
This is the machine learning type most frequently used in practice.

#### Gather Data  
The data is a collection of pairs (input, output). Input could be anything (emails, pictures, measurements) Outputs are usually real numbers, or labels (spam, not_spam, cat, dog, etc). Sometimes outputs are vectors (coordinates of a rectangle around a person in a picture), sequences (adjective, adjective, noun), or have some other structure.

In the email example, say you have 10,000 email messages each with a label of $spam$ or $not\_spam$. Now you have to convert each email into a feature vector.

One way to convert a text into a feature vector, called *bag of words*, is to take a dictionary (let's say it contains 20,000 words) and stipulate that in our feature vector:
- the first feature is equal to $1$ if the email contains the word "a"; otherwise, this feature is $0$;
- the second feature is equal to $1$ if the email contains the word "aaron"; otherwise this feature is $0$;
- ...
- the feature at position 20,000 is equal to $1$ if the email contains the word "zulu"; otherwise, this feature is equal to $0$

You repeat the above procedure for every email in our collection, which gives us 10,000 feature vectors(each having the dimensionality of 20,000) and a label ($spam$, $not\_spam$).

Now you have machine-readable input data, but the output labels are still in the form of human-readable text. Some learning algorithms require transforming labels into numbers (or booleans, functionally). Here, $spam$ could be $1$ (positive label) and $not\_spam$ could be $0$ (negative label). Support Vector Machine (SVM) requires the numeric value of $+1$ and $-1$.

#### Apply the Learning Algorithm
At this point, you have a dataset and a learning algorithm, so you are ready to apply the learning algorithm to the dataset to get the model.

SVM sees every feature vector as a point in a high-dimensional space. The algorithm puts all feature vectors on an imaginary 20,000-dimensional plot and draws an imaginary 19,999-dimensional line (a hyperplane) that separates examples with positive labels from examples with negative labels. The boundary separating the examples of different classes is called the decision boundary.

The equation of the hyperplane is given by two parameters, a real-valued vector $w$ of the same dimensionality as our input vector $x$, and a real number $b$ like this:
$$wx-b=0$$
where the expression $wx$ means:
$$w^{(1)}x^{(1)}+w^{(2)}x^{(2)}+...+w^{(D)}x^{(D)}$$
and $D$ is the number of dimensions of the feature vector $x$.

Now, the predicted label for some input feature vector $x$ is given:

$$y={\operatorname{sign}}(wx - b)$$

where sign is a mathematical operator that takes any value as input and returns $+1$ if the input is a positive number or $-1$ if the input is a negative number.

The goal of the learning algorithm - SVM in this case - is to leverage the dataset and find the optimal values of $w$* and $b$* for parameters $w$ and $b$. Once the learning algorithm identifies these optimal values, the model $f(x)$ is then defined as:

$$f(x)={\operatorname{sign}}(w^*x - b^*)$$

Therefore, to predict whether an email message is spam or not spam using an SVM model, you have to take the text of the message, convert it into a feature vector, then multiply this vector by $w$* , subtract $b$* and take the sign of the result. This will give us the prediction ($+1$ means $spam$, $-1$ means $not\_spam$).

The machine finds $w$* and $b$* by solving an optimization problem. Machines are good at optimizing functions under constraints. We want to satisfy a few constraints: first, we want the model to predict the labels of our 10,000 examples correctly. 

## Semi-supervised Learning
In semi-supervised learning, the dataset contains both labeled and unlabeled examples. Usually, there are much more unlabeled examples. The goal of a semi-supervised learning algorithm is the same as supervised. Additionally, there is the hope that using many unlabeled examples can help the learning algorithm to find a better model.

When you add unlabeled examples, you add more information about your problem, and a larger sample reflects better the probability distribution the data labeled came from.
## Unsupervised Learning
In unsupervised learning, the dataset is a collection of unlabeled examples: 
$$\{x_i\}^N_{i=1}$$
The goal of unsupervised learning algorithm is to create a model that takes a feature vector $x$ as input and either transform it into another vector or into a value that can be used to solve a practical problem.

| example  | output |
| ------------- | ------------- |
| clustering  | returns the id of the cluster for each feature vector in the dataset  |
| dimensionality reduction  | returns a feature vector that has fewer features than the input $x$ |
| outlier detection | returns a real number that indicates how $x$ is different from a typical example in the dataset |
## Reinforcement
Reinforcement learning is a subfield of machine learning where the machine exists in an environment and is capable of perceiving the state of that environment as a vector of features. The machine can execute actions in every state. Different actions bring different rewards and could also move the machine to another state of the environment. The goal of a reinforcement learning algorithm is to learn a policy.

A policy is a function(like a model in supervised learning) that takes the feature vector of a state as input and outputs an optimal action to execute in that state. The action is optimal *if* it maximizes the expected average reward.

Reinforcement learning solves a particular kind of problem where decision making is sequential, and the goal is long-term (game playing, robotics, resource management, logistics, etc). 
# Sources
[The Hundred-Page Machine Learning Book](https://themlbook.com/) by Andriy Burkov, 2019 