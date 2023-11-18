data {
  int<lower=0> N;  // number of observations
  int<lower=0> J;  // number of groups
  int<lower=1,upper=J> group[N];  // group indicator
  matrix[N, 3] X;  // predictor matrix
  int<lower=0,upper=1> y[N];  // response variable
}

parameters {
  real alpha[J];  // group intercepts
  vector[3] beta;  // fixed effects
  real<lower=0> sigma_alpha;  // sd of group intercepts
}

model {
  // Priors
  beta ~ normal(0, 1);
  alpha ~ normal(0, sigma_alpha);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(alpha[group[n]] + X[n] * beta);
  }
}
