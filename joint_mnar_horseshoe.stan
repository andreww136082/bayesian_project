data {
  int<lower=0> N;                 
  int<lower=0> P;               
  int<lower=0> K;           
  int<lower=0,upper=1> Y[N];      
  
  // Missing Data Architecture
  int<lower=0> N_obs;             
  int<lower=0> N_mis;             
  int<lower=1,upper=N> row_obs[N_obs]; 
  int<lower=1,upper=P> col_obs[N_obs]; 
  real X_obs[N_obs];              
  
  int<lower=1,upper=N> row_mis[N_mis]; 
  int<lower=1,upper=P> col_mis[N_mis]; 
  
  // Missingness Mechanism Data 
  int<lower=0,upper=1> M[N, K];   
}

transformed data {
  int<lower=0,upper=1> M_t[K, N];
  for (n in 1:N) {
    for (k in 1:K) {
      M_t[k, n] = M[n, k];
    }
  }
}

parameters {
  real beta0;
  vector[P] z;                    
  vector<lower=0>[P] lambda;      
  real<lower=0> tau;              
  
  vector[P] mu_x;                 
  vector<lower=0>[P] sigma_x;     
  
  real X_mis[N_mis];              
  
  // MNAR Parameters
  vector[K] xi0;                       
  vector[K] xi2;                       
}

transformed parameters {
  vector[P] beta = z .* lambda * tau;
  
  matrix[N, P] X_complete;
  for (i in 1:N_obs) {
    X_complete[row_obs[i], col_obs[i]] = X_obs[i];
  }
  for (i in 1:N_mis) {
    X_complete[row_mis[i], col_mis[i]] = X_mis[i];
  }
}

model {
  beta0 ~ normal(0, 5);
  z ~ std_normal();               
  lambda ~ cauchy(0, 1);          
  tau ~ cauchy(0, 1);             
  
  mu_x ~ normal(0, 5);
  sigma_x ~ cauchy(0, 2);
  
  xi0 ~ normal(0, 5);
  xi2 ~ normal(0, 1);             

  vector[N] Y_vec = to_vector(Y);
  
  // 1. Covariate Distribution 
  for (p in 1:P) {
    X_complete[, p] ~ normal(mu_x[p], sigma_x[p]);
  }
  
  // 2. MNAR Mechanism 
  for (k in 1:K) {
    M_t[k] ~ bernoulli_logit(xi0[k] + xi2[k] * Y_vec);
  }
  
  // 3. Outcome Model
  Y ~ bernoulli_logit_glm(X_complete, beta0, beta);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = bernoulli_logit_lpmf(Y[i] | beta0 + dot_product(X_complete[i], beta));
  }
}
