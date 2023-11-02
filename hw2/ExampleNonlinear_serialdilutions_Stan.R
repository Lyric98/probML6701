library(rstan)
library(tidybayes)
library(patchwork)
library(tidyverse)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

theme_set(
  theme_classic() +
    theme(
      panel.grid.major = element_line(),
      strip.background = element_blank(),
      plot.title = element_text(hjust = 0.5)
    )
)

SNS_BLUE <- "#1F77B4"
STAN_RED <- "#B2171D"

dilution_standards_data <- tibble::tribble(
  ~conc, ~dilution, ~y,
  0.64, 1, c(101.8, 121.4),
  0.32, 1 / 2, c(105.2, 114.1),
  0.16, 1 / 4, c(92.7, 93.3),
  0.08, 1 / 8, c(72.4, 61.1),
  0.04, 1 / 16, c(57.6, 50.0),
  0.02, 1 / 32, c(38.5, 35.1),
  0.01, 1 / 64, c(26.6, 25.0),
  0, 0, c(14.7, 14.2),
) %>%
  mutate(rep = purrr::map(conc, ~ c("a", "b"))) %>%
  unnest(c(y, rep))

knitr::kable(dilution_standards_data)

data_plot <- dilution_standards_data %>%
  ggplot(aes(x = conc, y = y, color = rep)) +
  geom_line(alpha = 0.5, linetype = 2) +
  geom_point(alpha = 0.8) +
  scale_x_continuous(expand = expansion(c(0, 0.02)), limits = c(0, NA)) +
  scale_y_continuous(expand = expansion(c(0, 0.02)), limits = c(0, NA)) +
  scale_color_brewer(type = "qual", palette = "Set1") +
  theme(
    legend.position = c(0.8, 0.2),
    legend.background = element_blank()
  ) +
  labs(
    x = "concentration",
    y = "y",
    title = "Serial dilution standard curve",
    color = "replicate"
  )

data_plot

library(mclust)
library(ggplot2)

# Extract the data
y <- dilution_standards_data$y

# Set up parameters
K <- 2  # Number of mixture components (assuming two replicates)
n <- length(y)
n_iter <- 1000  # Number of iterations

# Initialize
set.seed(123)
z <- sample(1:K, n, replace = TRUE)  # Component assignments
theta <- matrix(rnorm(2 * K), nrow = K)  # Means and variances [mu, sigma]
beta <- rep(1/K, K)  # Mixture proportions

# Storage for log-likelihood
log_likelihood <- numeric(n_iter)

# Gibbs sampling
for (iter in 1:n_iter) {
  # Sample z
  for (i in 1:n) {
    prob_z <- numeric(K)
    for (k in 1:K) {
      prob_z[k] <- dnorm(y[i], mean = theta[k, 1], sd = theta[k, 2]) * beta[k]
    }
    
    # Check and handle NA or NaN values in prob_z
    if (any(is.na(prob_z) | is.nan(prob_z)) || all(prob_z == 0)) {
      prob_z <- rep(1/K, K)
    }
    
    prob_z <- prob_z / sum(prob_z)
    z[i] <- sample(1:K, 1, prob = prob_z)
  }
  
  # Sample theta
  for (k in 1:K) {
    y_k <- y[z == k]
    if (length(y_k) > 1) { # Ensure we have more than one data point for a component
      theta[k, 1] <- mean(y_k)  # Update means
      theta[k, 2] <- sd(y_k)    # Update standard deviations
    }
  }
  
  # Sample beta
  beta <- table(z) / n
  
  # Calculate and store log-likelihood
  log_likelihood[iter] <- logLik(Mclust(y, G = K))
}

# Plot log-likelihood
ggplot(data.frame(iter = 1:n_iter, log_likelihood = log_likelihood), aes(x = iter, y = log_likelihood)) +
  geom_line() +
  labs(x = "Iteration", y = "Log-likelihood", title = "Convergence of Gibbs Sampling")


