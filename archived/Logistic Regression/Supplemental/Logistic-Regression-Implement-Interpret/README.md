# Logistic Regression 2

## Learning Goals

- explain how to interpret logistic regression coefficients
- describe different generalizations of linear regression for different scenarios

## Lecture Materials

[Jupyter Notebook: Logistic Regression](logistic_regression.ipynb)

## Lesson Plan

### Introduction (5 Mins)

We still need to talk about how to assess and to interpret the coefficients of logistic regression models.

### Log-Loss (15 Mins)

There is no closed-form solution to the logistic regression problem. Moreover, if we plug the sigmoid function into our loss function, we get a non-convex function. So we really need a new way of assessment. One way is with log-loss. Notice that the domain of the log-loss function is (0, 1), and so here we would need to use `.predict_proba()` to make a proper calculation. The code works through an example, also verifying `sklearn`'s calculation by hand.

### Betas (15 Mins)

The betas of a linear regression are pretty straightforwardly interpretable. The task is a bit more difficult for a logistic regression. We need to recall the idea of log-odds. We can also, just like in the case of linear regression, use the calculated betas to make a prediction with our logistic regression model.

### Multi-Class Classification (10 Mins)

Preparing the iris dataset for logistic regression modeling. The `sklearn` tool will use a "one-vs.-rest" strategy for the three species. The code walks through cross-val as well as accuracy calculations. (This metric will be explained soon but is quite intuitive.)

### Exercise (10 Mins)

The exercise asks students to apply logistic regression to the geysers dataset.

### Conclusion (5 Mins)

We've used log-loss to assess our models, but in fact there are many other classification metrics, each of which has its use cases. We'll turn to this next!