import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import os

from src.data_generator import DataGeneratorOutliers
from src.gp_models import get_gp
from src.train import optimize_gp
from src.plot import plot_points_1d, plot_gp_1d
from src.utils import schedule, filter_outliers, ei


def bo_with_outliers(data_generator:DataGeneratorOutliers, scheduler, p, percentile, lr=0.01, kernel="matern", niter=1000, seed=0, debug=False, **kwargs):
    
    torch.manual_seed(seed)
    
    x_test = data_generator.x
    y_test = data_generator.y
    
    # Initial points
    sample_x, sample_y, outliers = data_generator.initial_lhs(p)
    
    if debug:
        os.makedirs("tmp", exist_ok=True)
    
    # plot initial points
    if debug and data_generator.name == "forrester":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        plot_points_1d(sample_x, sample_y, outliers, x_test, y_test, ax=ax1)
        ax2.axis("off")
        fig.suptitle("Initial points")
        fig.tight_layout()
        fig.savefig(f"tmp/img000.png", dpi=300)

    # current_best_x = sample_x[torch.argmin(sample_y)]
    current_best_y = torch.min(sample_y)
    optimal_min = torch.min(y_test).item()
    best_y = [current_best_y.item()]
    error = np.inf
    
    for i, cond in enumerate(scheduler):
        print(f"Iteration {i:03}: schedule={cond}", end=" | " if not debug else "\n")
        if cond:
            if debug:
                print("Training GP with Student-T likelihood")
            model_st, likelihood_st, obj_func_st, optim_st, predict_st = get_gp("studentT", sample_x, sample_y, lr=lr, kernel=kernel)
            losses_st = optimize_gp(model_st, likelihood_st, obj_func_st, optim_st, sample_x, sample_y, n_iter=niter, silent=not debug)
            
            sample_x_filt, sample_y_filt = filter_outliers(model_st, likelihood_st, sample_x, sample_y, percentile=percentile)
            if debug:
                print(f"Filtered out {sample_x.size(0) - sample_x_filt.size(0)} outliers")
            
        if not cond or sample_y_filt.size(0) < sample_y.size(0) // 2:
            if debug:
                print("no filtering")
            sample_x_filt, sample_y_filt = sample_x.clone(), sample_y.clone()
        
        if debug:
            print("Training GP with Gaussian likelihood")
        model_g, likelihood_g, obj_func_g, optim_g, predict_g = get_gp("gaussian", sample_x_filt, sample_y_filt, lr=lr, kernel=kernel)
        losses_g = optimize_gp(model_g, likelihood_g, obj_func_g, optim_g, sample_x_filt, sample_y_filt, n_iter=niter, silent=not debug)
                
        eis = ei(x_test, model_g, likelihood_g, current_best_y)
        x_next = x_test[torch.argmax(eis)].unsqueeze(0)

        y_next, is_out = data_generator.generate(x_next)
        
        if debug and data_generator.name == "forrester":
            print("DEBUG: Plotting GP")
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
            
            plot_points_1d(sample_x, sample_y, torch.zeros_like(sample_y).bool(), x_test, y_test, ax=ax1)
            ax1.scatter(sample_x_filt, sample_y_filt, marker='o', c='red', label='filtered')
            ax1.scatter(x_next, y_next, marker='*', c='deeppink', label='next', s=200)
            
            gp_mean, gp_var = predict_g(model_g, likelihood_g, x_test)
            plot_gp_1d(x_test, gp_mean, gp_var, ax=ax1)
            
            ax2.plot(x_test, eis)
            ax2.set_xlabel("x")
            ax2.set_ylabel("EI")
            
            plt.suptitle(f"Iteration {i:03} - Current error: {error:.4f}")
            plt.tight_layout()
            plt.savefig(f"tmp/img{i+1:03}.png", dpi=300)
            plt.close()
        
        elif debug and data_generator.name == "branin":
            print("DEBUG: Plotting GP")
            fig = plt.figure(figsize=(12, 6))
            
            ax1= fig.add_subplot(121, projection='3d')
            ax1.plot_surface(x_test[..., 0].reshape((100, 100)), x_test[..., 1].reshape((100, 100)), y_test.reshape((100, 100)), cmap='coolwarm', alpha=.85)
            ax1.set_xlabel('x')
            ax1.set_xticks(np.arange(-5, 10.1, 5))
            ax1.set_ylabel('y')
            ax1.set_yticks(np.arange(0, 15.1, 5))
            ax1.set_zlabel('z')
            ax1.set_title("True function")
            
            ax2 = fig.add_subplot(122, projection='3d')
            gp_mean, gp_var = predict_g(model_g, likelihood_g, x_test)
            ax2.plot_surface(x_test[..., 0].reshape((100, 100)), x_test[..., 1].reshape((100, 100)), gp_mean.reshape((100, 100)), cmap='coolwarm', alpha=.85)
            ax2.set_xlabel('x')
            ax2.set_xticks(np.arange(-5, 10.1, 5))
            ax2.set_ylabel('y')
            ax2.set_yticks(np.arange(0, 15.1, 5))
            ax2.set_zlabel('z')
            ax2.set_title("GP mean")
            
            fig.tight_layout()
            plt.suptitle(f"Iteration {i:03} - Current error: {error:.4f}")
            plt.tight_layout()
            plt.savefig(f"tmp/img{i+1:03}.png", dpi=300)
            plt.close()
            
        
        current_best_y = torch.min(sample_y_filt)
        error = current_best_y.item() - optimal_min
        best_y.append(current_best_y.item())
        # if error < 1e-3:
        #     break
        
        sample_x = torch.cat([sample_x, x_next])
        sample_y = torch.cat([sample_y, y_next])
        
        print(f"Error = {error:.4f}") # x={current_best_x.item():.4f}, 
        # print()
    return np.array(best_y) - optimal_min, sample_x, sample_y

if __name__ == "__main__":
    SEED = 0
    gen_1d = DataGeneratorOutliers("branin", prob_outlier=.1, seed=SEED)
    
    max_t = 50
    init_t = 5
    every_t = 2
    scheduler = schedule(max_t, init_t, every_t)

    all_points = bo_with_outliers(gen_1d, scheduler, p=10, percentile=0.3, seed=SEED, debug=True, lr=0.05, niter=5000)
    print("done")