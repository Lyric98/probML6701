functions { real h(real x, real beta1, real beta2, real beta3, real beta4) {// Define our mean functions for y (machine reads) and x (concentration)
    real mean_function = beta1 + beta2/(1 + (x/beta3)^(-beta4));
    return mean_function;}}
data {
  int<lower=0> N_standard; // number of total reads of standard sample(# of dilution for each standard samples * # of standard samples)
  int<lower=0> N_unknown; // number of total reads of unknown sample(# of dilution for each unknown samples * # of unknown samples)
  int<lower=0> N_stand_sample;// number of standard samples (2 in our example)
  int<lower=0> N_unknown_sample;// number of unknown samples (10 in our example)
  vector [N_standard] y_standard; // vector of observed reads of standard samples 
  vector<lower=0> [N_standard] d_standard; // dilution for standard samples 
  int <lower=0> ind_standard[N_standard];
  vector [N_unknown] y_unknown;  // vector of observed reads of unknown samples 
  vector<lower=0> [N_unknown] d_unknown;  // dilution for unknown samples 
  int <lower=0> ind_unknown[N_unknown]; 
  vector<lower=0> [N_stand_sample] theta_0;//initial concnetration}
  int start_index[N_unknown_sample]; // use interger rather than real (vectors)to do the indexing
  int end_index[N_unknown_sample];
}
transformed  data{
  vector<lower=0>[N_standard] x_standard; // serial concentration for standard samples
  for(i in 1 : N_standard){ x_standard[i] = theta_0[ind_standard[i]] * d_standard[i];}
}
parameters {
  vector<lower=0>[N_unknown_sample] phi;
  vector<lower=0>[4] beta; 
  real<lower=0> sigma_y; 
  real<lower=0> mu_exp; 
  vector[N_unknown_sample] delta_beta2;
  vector<lower=0>[N_unknown_sample] delta_sigma;
  real<lower=0, upper = 1> lambda;
  }
transformed parameters{
  vector<lower=0>[N_unknown] x_unknown; // serial concentration for unknown samples 
  vector[N_unknown_sample] log_lik_A = rep_vector(0,N_unknown_sample);
  vector[N_unknown_sample] log_lik_B = rep_vector(0,N_unknown_sample);
  vector<lower=0>[N_unknown_sample] beta2_unknown;
  vector<lower=0>[N_unknown_sample] sigma_y_unknown;
  for(i in 1 : N_unknown) {x_unknown[i] = phi[ind_unknown[i]] * d_unknown[i];} 
  for(j in 1 : N_unknown_sample) {
    beta2_unknown[j] = beta[2]*exp(delta_beta2[j]);
    sigma_y_unknown[j] = sigma_y*exp(delta_sigma[j]);}
  for (j in 1:N_unknown_sample){
   for (k in start_index[j]:end_index[j]){
     log_lik_A[j] += normal_lpdf(y_unknown[k]|log(h(x_unknown[k], beta[1], beta[2], beta[3], beta[4])), sigma_y);
     log_lik_B[j] += normal_lpdf(y_unknown[k]|log(h(x_unknown[k], beta[1], beta2_unknown[j], beta[3], beta[4])), sigma_y_unknown[j]);
    }
  }
}
model {
  beta ~ lognormal(0, 10);
  sigma_y ~ normal(0, 10);
  mu_exp ~ normal(0,0.1); 
  phi ~ exponential(mu_exp); //hierarchical structure for unknown concentration
  delta_beta2 ~ normal(0,1);
  delta_sigma ~ exponential(1);
  lambda ~ beta(1,10);
  for (i in 1: N_standard){target += normal_lpdf(y_standard[i]|log(h(x_standard[i], beta[1], beta[2], beta[3], beta[4])), sigma_y);} 
  for (j in 1: N_unknown_sample){
    target += log_sum_exp(log(1-lambda) + log_lik_A[j], log(lambda) + log_lik_B[j]);
   }
} 
generated quantities {
  vector [N_unknown] y_tilde; // on log scale
  vector [N_unknown_sample] p_tilde; // probability that unknown sample is different from standard
  vector [N_unknown_sample] z_tilde; // underlying unobserved mixture indicator for each unknown sample
  vector [N_unknown] lp;
  vector [N_unknown] lp_plus;
  vector [N_unknown] lp_minus;
  vector [N_unknown] lp_mixture_latent_1;
  vector [N_unknown] lp_mixture_latent_2;
  for (j in 1: N_unknown_sample){
    p_tilde[j] = 1.0 / (1.0 + (1.0 - lambda) * exp(log_lik_A[j] - log_lik_B[j]) / lambda);
	z_tilde[j] = bernoulli_rng(p_tilde[j]);
    for(k in start_index[j]:end_index[j]){
      y_tilde[k] = (1-lambda)*normal_rng(log(h(x_unknown[k], beta[1], beta[2], beta[3], beta[4])), sigma_y) 
      + lambda*normal_rng(log(h(x_unknown[k], beta[1], beta2_unknown[j], beta[3], beta[4])), sigma_y_unknown[j]);
      lp[k] = log_sum_exp(log(1-lambda) + normal_lpdf(y_unknown[k]|log(h(x_unknown[k], beta[1], beta[2], beta[3], beta[4])), sigma_y), log(lambda) + normal_lpdf(y_unknown[k]|log(h(x_unknown[k], beta[1], beta2_unknown[j], beta[3], beta[4])), sigma_y_unknown[j]));
      lp_mixture_latent_1[k] = normal_lpdf(y_unknown[k]|log(h(x_unknown[k], beta[1], beta[2], beta[3], beta[4])), sigma_y);
      lp_mixture_latent_2[k] = normal_lpdf(y_unknown[k]|log(h(x_unknown[k], beta[1], beta2_unknown[j], beta[3], beta[4])), sigma_y_unknown[j]);
      lp_minus[k] = log_sum_exp(log(1-lambda) + normal_lpdf(y_unknown[k] - 0.1|log(h(x_unknown[k], beta[1], beta[2], beta[3], beta[4])), sigma_y), log(lambda) + normal_lpdf(y_unknown[k] - 0.1|log(h(x_unknown[k], beta[1], beta2_unknown[j], beta[3], beta[4])), sigma_y_unknown[j]));
      lp_plus[k] = log_sum_exp(log(1-lambda) + normal_lpdf(y_unknown[k] + 0.1|log(h(x_unknown[k], beta[1], beta[2], beta[3], beta[4])), sigma_y), log(lambda) + normal_lpdf(y_unknown[k] + 0.1|log(h(x_unknown[k], beta[1], beta2_unknown[j], beta[3], beta[4])), sigma_y_unknown[j]));
    }
  }
}
