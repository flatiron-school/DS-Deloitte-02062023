# Logistic Regression 1

## Learning Goals

- describe conceptually the need to move beyond linear regression;
- explain the relationship between probability and odds;
- explain the form of logistic regression;

## Lecture Materials

[Jupyter Notebook: Logistic Regression](logistic_regression.ipynb)

## Lesson Plan

### Introduction (5 Mins)

Intro to classification models! New problems solvable now (where the target is a category).

### Using Linear Regression (15 Mins)

Attempting a classification problem with linear regression.

### Sigmoid Function (10 Mins)

Structure of logistic regression. Idea of using a link function. Sigmoid (or "expit") as smashing all values between 0 and 1, which is what we want for a classification problem. (We never get 0 or 1 exactly, but we can round to the nearer value.)

### Logit Function: Odds and Probability (10 Mins)

The link function for logistic regression (ln(y / (1-y))) is the logit function, and it's the inverse of sigmoid.

### `LogisticRegression()` (5 Mins)

The `sklearn` API for logistic regression works much like that for linear regression. There are some parameters that we'll talk about in the next lecture.

### `.predict()` vs. `.predict_proba()` (10 Mins)

If we just want the rounded, final prediction, we'll use `.predict()`. But sometimes we want the percentages, and so we'll use `.predict_proba()`.

### Conclusion (5 Mins)

Some remaining questions! What does evaluation of classification models look like? How do we interpret the betas of a logistic regression model? Etc.