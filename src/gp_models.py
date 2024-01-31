import gpytorch
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, kernel="matern", **kwargs):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "matern":
            # print("Using Matern kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(**kwargs)
            )
        elif kernel == "rbf":
            # print("Using RBF kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(**kwargs)
            )
        else:
            raise ValueError("Unknown kernel name: {}".format(kernel))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel="matern", **kwargs):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "matern":
            # print("Using Matern kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(**kwargs)
            )
        elif kernel == "rbf":
            # print("Using RBF kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(**kwargs)
            )
        else:
            raise ValueError("Unknown kernel name: {}".format(kernel))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_gp(name, x_train, y_train, lr=0.01, **kwargs):
    if name == "gaussian":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x_train, y_train, likelihood, **kwargs)

        objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)

        def predict(model, likelihood, x):
            model.eval()
            likelihood.eval()

            with torch.no_grad():
                observed_pred = likelihood(model(x))
                # samples = observed_pred.sample(sample_shape=x.size())
            return observed_pred.mean, observed_pred.variance

    elif name == "studentT":
        likelihood = gpytorch.likelihoods.StudentTLikelihood()
        model = ApproximateGPModel(x_train, **kwargs).to(dtype=torch.float64)

        objective = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=len(y_train)
        )
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=lr
        )

        def predict(model, likelihood, x):
            model.eval()
            likelihood.eval()

            with torch.no_grad():
                observed_pred = likelihood(model(x))
                # samples = observed_pred.sample(sample_shape=x.size())
            return observed_pred.mean, observed_pred.variance

    else:
        raise ValueError("Unknown GP name")

    return model, likelihood, objective, optimizer, predict
