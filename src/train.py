import torch
import gpytorch
from tqdm import tqdm

def optimize_gp(model, likelihood, obj_func, optimizer, x_train, y_train, n_iter=50, silent=False):
    # Set the model and likelihood to training mode
    model.train()
    likelihood.train()

    iter_verbose = n_iter // 10
    record_loss = []

    with tqdm(range(0, n_iter + 1), ncols=150, disable=silent) as pbar:
        
        for i in pbar:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train)
            # Compute loss and backprop gradients
            loss = -obj_func(output, y_train)
            record_loss.append(loss.item())
            loss.backward()
            if i % iter_verbose == 0:
                pbar.set_postfix({"loss": loss.item()})
            optimizer.step()

    return record_loss


def em(x, y, f, kernel_mat, nu, sigma, n_iter=100, eps_conv=1e-3):
    n = len(x)
    compute_inv_v = lambda f, y: torch.diag((nu + 1) / (nu * sigma**2 + (f - y) ** 2))
    errors = []

    for _ in range(n_iter):
        f_old = f.clone()
        inv_v = compute_inv_v(f, y)

        # naive implementation
        # f = (kernel_mat.inverse() + inv_v).inverse() @ inv_v @ y

        # more stable one
        inv_v_half = inv_v**0.5
        b = torch.eye(n) + (inv_v_half) @ kernel_mat @ (inv_v_half)
        f = (
            (
                kernel_mat
                - kernel_mat @ inv_v_half @ b.inverse() @ inv_v_half @ kernel_mat
            )
            @ inv_v
            @ y
        )

        error = torch.linalg.norm(f - f_old).item()
        errors.append(error)

        if error < eps_conv:
            break

    return f, errors
