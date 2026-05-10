import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Tell JAX to use the NVIDIA GPU
numpyro.set_platform("cuda")
numpyro.set_host_device_count(1)

def run_pipeline():
    # Grab Slurm Array ID (Chain) and Tau Scale
    chain_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tau_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    data_dir = "r_bayesian_data"
    
    X_fused = np.load(f"{data_dir}/train_X_embeddings.npy").astype(np.float32)
    Y_labels = np.load(f"{data_dir}/train_Y_labels.npy").astype(np.int32)
    M_mod = np.load(f"{data_dir}/train_M_indicators.npy").astype(np.int32)

    Y = Y_labels[:, 0] if Y_labels.ndim == 2 else Y_labels

    np.random.seed(42)
    sample_idx = np.random.choice(X_fused.shape[0], 1000, replace=False)
    X_fused, Y, M_mod = X_fused[sample_idx], Y[sample_idx], M_mod[sample_idx]
    
    X = X_fused
    N, P = X.shape
    K = M_mod.shape[1]
    dim_per_mod = P // K

    M_expanded = np.repeat(M_mod, dim_per_mod, axis=1)

    obs_idx = np.where(M_expanded == 1)
    mis_idx = np.where(M_expanded == 0)
    
    X_obs_values = X[obs_idx]
    
    M_t = M_mod.T

    def joint_mnar_horseshoe(X_obs, obs_idx, mis_idx, M_t, Y, N, P, K, tau_scale):
        beta0 = numpyro.sample("beta0", dist.Normal(0, 5))
        with numpyro.plate("features", P):
            z = numpyro.sample("z", dist.Normal(0, 1))
            lambda_ = numpyro.sample("lambda", dist.HalfCauchy(1))
            
        tau = numpyro.sample("tau", dist.HalfCauchy(tau_scale))
        
        beta = z * lambda_ * tau

        with numpyro.plate("cov_params", P):
            mu_x = numpyro.sample("mu_x", dist.Normal(0, 5))
            sigma_x = numpyro.sample("sigma_x", dist.HalfCauchy(2))

        X_mis = numpyro.sample("X_mis", dist.Normal(mu_x[mis_idx[1]], sigma_x[mis_idx[1]]))
        numpyro.sample("X_obs", dist.Normal(mu_x[obs_idx[1]], sigma_x[obs_idx[1]]), obs=X_obs)

        X_complete = jnp.zeros((N, P))
        X_complete = X_complete.at[obs_idx].set(X_obs)
        X_complete = X_complete.at[mis_idx].set(X_mis)

        with numpyro.plate("modalities", K):
            xi0 = numpyro.sample("xi0", dist.Normal(0, 5))
            xi2 = numpyro.sample("xi2", dist.Normal(0, 1))
            
        logits_m = xi0[:, None] + xi2[:, None] * Y[None, :]
        numpyro.sample("M_lik", dist.Bernoulli(logits=logits_m), obs=M_t)

        logits_y = beta0 + jnp.dot(X_complete, beta)
        numpyro.sample("Y_lik", dist.Bernoulli(logits=logits_y), obs=Y)

    
    nuts_kernel = NUTS(joint_mnar_horseshoe, 
                       target_accept_prob=0.90, 
                       max_tree_depth=10,
                       init_strategy=numpyro.infer.init_to_median())
    
    mcmc = MCMC(nuts_kernel, 
                num_warmup=2500, 
                num_samples=2500, 
                num_chains=1,
                progress_bar=True) 

    mcmc.run(jax.random.PRNGKey(42 + chain_id), 
             X_obs_values, obs_idx, mis_idx, M_t, Y, N, P, K, tau_scale)

    print("Sampling complete. Saving arrays...")
    samples = mcmc.get_samples()
    
    # Format: numpyro_tau_1.0_chain_1.npz
    out_file = f"numpyro_tau_{tau_scale}_chain_{chain_id}.npz"
    np.savez_compressed(out_file, **samples)
    print(f"Data successfully saved to {out_file}.")

if __name__ == "__main__":
    run_pipeline()