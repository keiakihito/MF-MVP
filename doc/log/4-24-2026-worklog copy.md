# Worklog — April 24th (Fri)

## Archive

- Understand quation 1
Matrix Factorization Pipeline (Intuitive Summary)
1. Input:　Start with an observed (sparse) interaction matrix R (item x user).

2.　Training: Learn two dense matrices P and Q using only the observed entries in R, such that:

R ~ PQ^T

3. Reconstruction: Since P and Q are dense, computing

PQ^T = R^ 

produces a fully dense matrix, where previously unobserved entries now have values. The unknown score is filled up, which is prediction score for the user.

4. Prediction:These values in the unobserved positions of 
R^ are the predicted scores in all users and all items.

## Action Plan
- Read paper and understand the codebase.

