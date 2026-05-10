library(reticulate)
library(ggplot2)
library(reshape2)
library(RcppCNPy)

np <- import("numpy")


tau_scale <- "0.01" 
num_chains <- 4
iter_per_chain <- 2500
calc_rhat <- function(chain_matrix) {
  # chain_matrix: rows = iterations, cols = chains
  n <- nrow(chain_matrix)
  m <- ncol(chain_matrix)
  
  chain_means <- colMeans(chain_matrix)
  chain_vars <- apply(chain_matrix, 2, var)
  overall_mean <- mean(chain_matrix)
  
  B <- (n / (m - 1)) * sum((chain_means - overall_mean)^2)
  W <- mean(chain_vars)
  
  V_hat <- ((n - 1) / n) * W + (1 / n) * B
  return(sqrt(V_hat / W))
}

beta0_mat <- matrix(NA, nrow = iter_per_chain, ncol = num_chains)
tau_mat <- matrix(NA, nrow = iter_per_chain, ncol = num_chains)
xi2_mat <- matrix(NA, nrow = iter_per_chain, ncol = num_chains) 

beta_chains_list <- list()

for (c in 1:num_chains) {
  filename <- sprintf("numpyro_tau_%s_chain_%d.npz", tau_scale, c)
  if (file.exists(filename)) {
    samples <- np$load(filename)
    
    beta0_mat[, c] <- samples[["beta0"]]
    tau_mat[, c] <- as.numeric(samples[["tau"]])
    
    # xi2 is shape (2500, K). We grab the sensitivity of the first modality [ , 1]
    xi2_mat[, c] <- samples[["xi2"]][, 1] 
    
    # Reconstruct beta
    z_mat <- samples[["z"]]            
    lambda_mat <- samples[["lambda"]]  
    tau_vec <- as.numeric(samples[["tau"]]) 
    beta_reconstructed <- sweep(z_mat * lambda_mat, 1, tau_vec, "*")
    
    beta_chains_list[[c]] <- beta_reconstructed
  } else {
    stop(sprintf("Could not find %s", filename))
  }
}

# Identify the single most active feature across all chains
stacked_beta <- do.call(rbind, beta_chains_list)
mean_abs_beta <- colMeans(abs(stacked_beta))
active_idx <- which.max(mean_abs_beta)

active_beta_mat <- matrix(NA, nrow = iter_per_chain, ncol = num_chains)
for (c in 1:num_chains) {
  active_beta_mat[, c] <- beta_chains_list[[c]][, active_idx]
}

create_trace_df <- function(mat, param_name) {
  df <- melt(mat)
  colnames(df) <- c("Iteration", "Chain", "Value")
  df$Chain <- as.factor(df$Chain)
  df$Parameter <- param_name
  return(df)
}

df_trace <- rbind(
  create_trace_df(beta0_mat, "Intercept (beta0)"),
  create_trace_df(tau_mat, "Global Shrinkage (tau)"),
  create_trace_df(active_beta_mat, sprintf("Active Feature (beta_%d)", active_idx))
)

p_trace <- ggplot(df_trace, aes(x = Iteration, y = Value, color = Chain)) +
  geom_line(alpha = 0.7, size = 0.3) +
  facet_wrap(~Parameter, scales = "free_y", ncol = 1) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom") +
  labs(title = "HMC Traceplots for Key Parameters", x = "Post-Warmup Iteration")

ggsave("mcmc_traceplots.pdf", plot = p_trace, width = 8, height = 8)

df_xi2 <- data.frame(Value = as.vector(xi2_mat))
mean_xi2 <- mean(df_xi2$Value)
ci_lower <- quantile(df_xi2$Value, 0.025)
ci_upper <- quantile(df_xi2$Value, 0.975)

p_xi2 <- ggplot(df_xi2, aes(x = Value)) +
  geom_density(fill = "steelblue", alpha = 0.5, color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = mean_xi2, color = "black", size = 1) +
  theme_minimal(base_size = 14) +
  labs(title = bquote("Posterior Density of MNAR Parameter (" ~ xi[2] ~ ")"),
       subtitle = sprintf("Mean: %.2f | 95%% CI: [%.2f, %.2f]", mean_xi2, ci_lower, ci_upper),
       x = bquote(xi[2] ~ " Value"), y = "Density") +
  annotate("text", x = 0, y = 0.1, label = "MAR Null Hypothesis", color = "red", angle = 90, vjust = -0.5)

ggsave("mnar_sensitivity_xi2.pdf", plot = p_xi2, width = 7, height = 5)

data_dir <- "r_bayesian_data"
X_test_fused <- npyLoad(file.path(data_dir, "test_X_embeddings.npy"))
Y_test_labels <- npyLoad(file.path(data_dir, "test_Y_labels.npy"))
if (is.matrix(Y_test_labels)) {
  Y_test <- as.numeric(Y_test_labels[, 1])
} else {
  Y_test <- as.numeric(Y_test_labels) 
}
M_test_mod <- npyLoad(file.path(data_dir, "test_M_indicators.npy"))
K <- ncol(M_test_mod)
P <- ncol(X_test_fused)
dim_per_mod <- P / K 
M_test <- M_test_mod[, rep(1:K, each = dim_per_mod)]
X_test <- X_test_fused
X_test[M_test == 0] <- NA
X_test_imp <- apply(X_test, 2, function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); return(x) })

posterior_beta <- apply(stacked_beta, 2, median)
posterior_beta0 <- median(beta0_mat)

logits <- posterior_beta0 + (X_test_imp %*% posterior_beta)
preds <- 1 / (1 + exp(-logits))

df_ppc <- data.frame(
  Prediction = as.numeric(preds),
  Actual = as.factor(ifelse(Y_test == 1, "Mortality (1)", "Survival (0)"))
)

p_ppc <- ggplot(df_ppc, aes(x = Prediction, fill = Actual)) +
  geom_density(alpha = 0.6) +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = c("Mortality (1)" = "firebrick", "Survival (0)" = "dodgerblue")) +
  labs(title = "Posterior Predictive Density by Actual Outcome",
       x = "Predicted Probability of Mortality",
       y = "Density", fill = "Observed Outcome") +
  theme(legend.position = "bottom")

ggsave("ppc_density.pdf", plot = p_ppc, width = 7, height = 5)
