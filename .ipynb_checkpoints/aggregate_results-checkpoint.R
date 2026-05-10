library(glmnet)
library(mice)
library(pROC)
library(PRROC)
library(RcppCNPy)
library(knitr) 

results_df <- data.frame(
  Model = character(),
  AUROC = numeric(),
  AUPRC = numeric(),
  Active_Features = character(),
  Notes = character(),
  stringsAsFactors = FALSE
)
data_dir <- "r_bayesian_data"
cat("Loading test embeddings...\n")
X_test_fused <- npyLoad(file.path(data_dir, "test_X_embeddings.npy"))
Y_test_labels <- npyLoad(file.path(data_dir, "test_Y_labels.npy"))
M_test_mod <- npyLoad(file.path(data_dir, "test_M_indicators.npy"))

if (is.matrix(Y_test_labels)) {
  Y_test <- as.numeric(Y_test_labels[, 1])
} else {
  Y_test <- as.numeric(Y_test_labels) 
}

N_test <- nrow(X_test_fused)
P <- ncol(X_test_fused)
K <- ncol(M_test_mod)
dim_per_mod <- P / K 

# Recreate the element-wise missingness mask
M_test <- M_test_mod[, rep(1:K, each = dim_per_mod)]
X_test <- X_test_fused
X_test[M_test == 0] <- NA
colnames(X_test) <- paste0("Cov_", 1:P)

cat(sprintf("Test set loaded: N = %d, P = %d\n\n", N_test, P))
if (file.exists("baseline_mice_results.rds")) {
  cat("--- Evaluating MICE + GLM ---\n")
  pooled_mice <- readRDS("baseline_mice_results.rds")
  mice_summary <- summary(pooled_mice)
  
  na_coefs <- sum(is.na(mice_summary$estimate))
  max_se <- max(mice_summary$std.error, na.rm = TRUE)
  
  cat(sprintf("MICE Statistical Collapse Diagnostics:\n"))
  cat(sprintf(" -> Dropped Coefficients (NA): %d out of %d\n", na_coefs, P))
  cat(sprintf(" -> Maximum Standard Error: %e\n", max_se))
  
  beta_mice <- mice_summary$estimate
  beta_mice[is.na(beta_mice)] <- 0 
  intercept <- beta_mice[1]
  weights <- beta_mice[-1]
  
  X_test_imp <- apply(X_test, 2, function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); return(x) })
  
  logits_mice <- intercept + (X_test_imp %*% weights)
  preds_mice <- 1 / (1 + exp(-logits_mice))
  
  roc_mice <- roc(Y_test, as.numeric(preds_mice), quiet = TRUE)
  pr_mice <- pr.curve(scores.class0 = preds_mice[Y_test == 1], 
                      scores.class1 = preds_mice[Y_test == 0], curve = FALSE)
  
  cat(sprintf("MICE Out-of-Sample AUROC: %.4f\n", roc_mice$auc))
  cat(sprintf("MICE Out-of-Sample AUPRC: %.4f\n\n", pr_mice$auc.integral))
  
  # Append to results
  results_df <- rbind(results_df, data.frame(
    Model = "MICE + GLM",
    AUROC = roc_mice$auc,
    AUPRC = pr_mice$auc.integral,
    Active_Features = as.character(P - na_coefs),
    Notes = sprintf("Rank Collapse (%d dropped)", na_coefs)
  ))
}

if (file.exists("baseline_lasso_results.rds")) {
  cat("--- Evaluating Frequentist Lasso ---\n")
  cv_lasso <- readRDS("baseline_lasso_results.rds")
  
  X_test_med <- apply(X_test, 2, function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); return(x) })
  preds_lasso <- predict(cv_lasso, newx = X_test_med, type = "response", s = "lambda.min")
  
  active_coefs <- sum(coef(cv_lasso, s = "lambda.min") != 0)
  cat(sprintf("Lasso Active Features Spared: %d out of %d\n", active_coefs, P))
  
  roc_lasso <- roc(Y_test, as.numeric(preds_lasso), quiet = TRUE)
  pr_lasso <- pr.curve(scores.class0 = preds_lasso[Y_test == 1], 
                       scores.class1 = preds_lasso[Y_test == 0], curve = FALSE)
  
  cat(sprintf("Lasso Out-of-Sample AUROC: %.4f\n", roc_lasso$auc))
  cat(sprintf("Lasso Out-of-Sample AUPRC: %.4f\n\n", pr_lasso$auc.integral))
  
  results_df <- rbind(results_df, data.frame(
    Model = "Lasso (Median Imp)",
    AUROC = roc_lasso$auc,
    AUPRC = pr_lasso$auc.integral,
    Active_Features = as.character(active_coefs),
    Notes = "Frequentist Baseline"
  ))
}

if (file.exists("baseline_cc_results.rds")) {
  cat("--- Evaluating Complete Case Analysis ---\n")
  fit_cc <- readRDS("baseline_cc_results.rds")
  
  cc_test_idx <- complete.cases(X_test)
  X_test_cc <- X_test[cc_test_idx, , drop = FALSE]
  Y_test_cc <- Y_test[cc_test_idx]
  
  cat(sprintf("Test patients retained for Complete Case: %d out of %d\n", sum(cc_test_idx), N_test))
  
  if(sum(cc_test_idx) > 0) {
    df_test_cc <- as.data.frame(X_test_cc)
    colnames(df_test_cc) <- paste0("Cov_", 1:P)
    
    suppressWarnings({
      preds_cc <- predict(fit_cc, newdata = df_test_cc, type = "response")
    })
    
    preds_cc[is.na(preds_cc)] <- 0.5 
    
    roc_cc <- roc(Y_test_cc, as.numeric(preds_cc), quiet = TRUE)
    pr_cc <- pr.curve(scores.class0 = preds_cc[Y_test_cc == 1], 
                      scores.class1 = preds_cc[Y_test_cc == 0], curve = FALSE)
    
    cat(sprintf("Complete Case AUROC: %.4f\n", roc_cc$auc))
    cat(sprintf("Complete Case AUPRC: %.4f\n", pr_cc$auc.integral))
    
    # Append to results
    results_df <- rbind(results_df, data.frame(
      Model = "Complete Case",
      AUROC = roc_cc$auc,
      AUPRC = pr_cc$auc.integral,
      Active_Features = "All (Rank Deficient)",
      Notes = sprintf("N=%d retained", sum(cc_test_idx))
    ))
  }
}

library(reticulate)
np <- import("numpy")

X_test_imp <- apply(X_test, 2, function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); return(x) })
tau_scales <- c("0.01", "1.0", "10.0")

for (scale in tau_scales) {
  beta_chains <- list()
  beta0_chains <- list()
  
  for (chain in 1:4) {
    filename <- sprintf("numpyro_tau_%s_chain_%d.npz", scale, chain)
    
    if (file.exists(filename)) {
      samples <- np$load(filename)
      
      z_mat <- samples[["z"]]           
      lambda_mat <- samples[["lambda"]] 
      tau_vec <- as.numeric(samples[["tau"]]) 
      
      beta_reconstructed <- sweep(z_mat * lambda_mat, 1, tau_vec, "*")
      
      beta_chains[[chain]] <- beta_reconstructed
      beta0_chains[[chain]] <- samples[["beta0"]]
    }
  }  
  
  if (length(beta_chains) > 0) {
    
    stacked_beta <- do.call(rbind, beta_chains)
    stacked_beta0 <- unlist(beta0_chains)
    
    posterior_beta <- apply(stacked_beta, 2, median)
    posterior_beta0 <- median(stacked_beta0)
    
    active_features <- sum(abs(posterior_beta) > 0.01)
    
    logits <- posterior_beta0 + (X_test_imp %*% posterior_beta)
    preds <- 1 / (1 + exp(-logits))
    
    roc_obj <- roc(Y_test, as.numeric(preds), quiet = TRUE)
    pr_obj <- pr.curve(scores.class0 = preds[Y_test == 1], 
                       scores.class1 = preds[Y_test == 0], curve = FALSE)
    
    # Append to results
    results_df <- rbind(results_df, data.frame(
      Model = sprintf("Bayesian Horseshoe (τ=%.2f)", as.numeric(scale)),
      AUROC = roc_obj$auc,
      AUPRC = pr_obj$auc.integral,
      Active_Features = as.character(active_features),
      Notes = sprintf("%d chains pooled", length(beta_chains))
    ))
  }
}


# Write to CSV
csv_file <- "evaluation_metrics.csv"
write.csv(results_df, csv_file, row.names = FALSE)

# Write to Markdown
md_file <- "evaluation_metrics.md"
md_table <- kable(results_df, format = "markdown", digits = 4)
writeLines(md_table, md_file)

# Write to LaTeX
tex_file <- "evaluation_metrics.tex"
tex_table <- kable(results_df, format = "latex", booktabs = TRUE, digits = 4, 
                   caption = "Out-of-Sample Evaluation Metrics on Held-Out Test Set",
                   label = "tab:results")
writeLines(tex_table, tex_file)
