Bayes Formula All terms explicit equation for MNIST Dataset -

P(Y|X) * P(Y) = P(X|Y) * P(X) => Bayes Formula

We here in the code compute the explicit equations of each of the above term for MNIST Dataset modelled via Autoencoder with Chebyshev Polynomial Approximation
of Relu (activation function used for every layer in the considered Autoencoder) with Chebyshev Nodes, but other activations can also be used and approximated by this 
approach , more of which is discussed in our deterministic Interpretation of NN paper.
Weights and biases can also be taken in Symbolic Notation and hence Prior on them can be set Via Bayesian Stats and using coordinate Ascent (15th Lecture of Andrew NG)
or using the Max Likelihood for finding the parameters [W1,W2,W3,Wd1,Wd2,Wd3,b1,b2,b3,bd1,bd2,bd3] 
explicitly without relying on the heuristic loss function landscape minimization , but
Quintic equations itself are not solvable , hence we cannot get analytic solutions
of weights and biases using 'Deterministic interpretation of NN' , but we can get 
log likelihood equation instead of heuristic loss function and now we can maximize that
or minimize the negative log likehood using the conventional optimization algorithms of ML and DL...

This study also calls for the Need for Symbolic Computation Libraries Research for faster compute , faster simplifying the expressions and faster operations on these expressions
So That P(X_hat|X) , P(Y_hat|Y) , P(X,Y) can be computed
Also the likelihood Computation and finding the exponential family plus the residual family of the dataset via 
symbolic libraries needs faster libraries or more compute for faster exececution and simplication and symbolic operation
on the symbolic expressions gotten via Chebyshev Activation Approximation of Relu via NN Model...


