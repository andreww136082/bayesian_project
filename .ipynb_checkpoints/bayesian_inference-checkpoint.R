library(rstan)
library(glmnet)
library(mice)
# Read numpy arrays natively
library(RcppCNPy)
args <- commandArgs(trailingOnly = TRUE)
chain_id <- ifelse(length(args) > 0, as.integer(args[1]), 1)

data_dir <- "r_bayesian_data"
X_fused <- npyLoad(file.path(data_dir, "train_X_embeddings.npy"))
Y_labels <- npyLoad(file.path(data_dir, "train_Y_labels.npy"))
M_mod <- npyLoad(file.path(data_dir, "train_M_indicators.npy"))

if (is.matrix(Y_labels)) {
  Y <- as.numeric(Y_labels[, 1])
} else {
  Y <- as.numeric(Y_labels) 
}

set.seed(42)
N_full <- nrow(X_fused)
sample_idx <- sample(1:N_full, 500)

X_fused <- X_fused[sample_idx, ]
Y <- Y[sample_idx]
M_mod <- M_mod[sample_idx, ]
# ---------------------------------------------------------

N <- nrow(X_fused)
P <- ncol(X_fused)
K <- ncol(M_mod)

# Calculate dimensions per modality (e.g., 1280 / 5 = 256)
dim_per_mod <- P / K 

# Create the expanded mask strictly for NA injection into X
M_expanded <- M_mod[, rep(1:K, each = dim_per_mod)]

X <- X_fused
X[M_expanded == 0] <- NA

obs_indices <- which(M_expanded == 1, arr.ind = TRUE)
mis_indices <- which(M_expanded == 0, arr.ind = TRUE)

stan_data <- list(
  N = N, P = P, K = K, 
  Y = as.integer(Y), 
  N_obs = nrow(obs_indices),
  N_mis = nrow(mis_indices),
  row_obs = obs_indices[, 1], 
  col_obs = obs_indices[, 2],
  X_obs = as.numeric(X[obs_indices]),
  row_mis = mis_indices[, 1], 
  col_mis = mis_indices[, 2],
  
  # Pass the UNEXPANDED N x K modality matrix to Stan
  M = matrix(as.integer(M_mod), nrow = N, ncol = K)
)


if (chain_id == 1) {
  cat("Running Baselines...\n")
  
  complete_cases <- complete.cases(X)
  
  # Build a proper dataframe so glm() binds to column names, not the matrix object
  df_cc <- data.frame(Y = Y[complete_cases], X[complete_cases, , drop = FALSE])
  colnames(df_cc)[-1] <- paste0("Cov_", 1:P) # Name the features Cov_1 to Cov_P
  
  fit_cc <- glm(Y ~ ., data = df_cc, family = binomial())
  saveRDS(fit_cc, "baseline_cc_results.rds")
    
  
  # # Lasso (Median Imputation)
  X_med <- apply(X, 2, function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); return(x) })
  cv_lasso <- cv.glmnet(X_med, Y, family = "binomial", alpha = 1)
  saveRDS(cv_lasso, "baseline_lasso_results.rds")
  
  cat("Running MICE imputation...\n")

    # 1. Assign explicit column names to X so we can build a formula
    colnames(X) <- paste0("Cov_", 1:P)
    df_mice <- data.frame(Y = Y, X)
    
    # 2. Run MICE
    mice_imp <- mice(df_mice, m = 3, method = "norm.predict", printFlag = FALSE)
    
    # 3. Explicitly construct the formula string
    cat("Fitting GLMs to imputed datasets...\n")
    models <- lapply(1:3, function(i) {
      dat <- complete(mice_imp, i)
      glm(Y ~ ., data = dat, family = binomial())
    })
    
    # 4. Convert back to a MICE analysis object and pool the results
    fit_mice <- as.mira(models)
    saveRDS(pool(fit_mice), "baseline_mice_results.rds")
    cat("MICE baseline saved.\n")
}

cat(sprintf("Starting Stan HMC for Chain %d...\n", chain_id))
fit_stan_single <- stan(
  file = "joint_mnar_horseshoe.stan",
  data = stan_data,
  iter = 5000,        
  warmup = 2500,
  thin = 2,           
  chains = 1,          
  chain_id = chain_id, 
  control = list(adapt_delta = 0.90, max_treedepth = 10),
  seed = 42 + chain_id,
  init = 0,            
  
  refresh = 10,
  
  pars = c("X_mis", "z", "mu_x", "sigma_x", "log_lik"), 
  include = FALSE
)

saveRDS(fit_stan_single, sprintf("stan_chain_%d.rds", chain_id))
cat(sprintf("Chain %d completed successfully.\n", chain_id))