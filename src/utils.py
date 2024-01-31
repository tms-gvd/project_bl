import torch
import scipy.stats as st

def schedule(max_t, init, every_step):
    for t in range(max_t):
        yield (t > init) and (t % every_step == 0)


def filter_outliers(model, likelihood, x, y, percentile=0.05):
    with torch.no_grad():
        y_pred = likelihood(model(x))
        y_mean = y_pred.mean.mean(0)
        y_std = y_pred.stddev.mean(0)
        z_score = (y - y_mean) / y_std
        bool_outlier = (z_score < st.norm.ppf(percentile)) | (z_score > st.norm.ppf(1 - percentile))
        x_filt = x[~bool_outlier]
        y_filt = y[~bool_outlier]
    return x_filt, y_filt

def ei(x, model, likelihood, current_best):
    
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_preds = likelihood(model(x))
        post_means, post_vars = observed_preds.mean, observed_preds.variance

    z = (current_best - post_means) / torch.sqrt(post_vars)

    return (current_best - post_means) * torch.distributions.normal.Normal(0, 1).cdf(z) + torch.sqrt(post_vars) * torch.exp(torch.distributions.normal.Normal(0, 1).log_prob(z))
