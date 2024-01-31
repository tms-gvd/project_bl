import torch
import lhsmdu

def forrester_function(x):
    return (6 * x - 2) ** 2 * torch.sin(12 * x - 4)


def branin_function(x, l):
    x0 = x[..., 0]
    x1 = x[..., 1]
    return (
        (
            x1
            - (5.1 / (4 * torch.pi**2) - 0.1 * (1 - l)) * x0 ** 2
            + 5 * x0 / torch.pi
            - 6
        )** 2
        + 10 * (1 - 1 / (8 * torch.pi)) * torch.cos(x0)
        + 10
    )
    
class DataGeneratorOutliers:
    
    def __init__(self, name, true_func=None, boundaries=None, prob_outlier=.05, alpha_reg=1., alpha_out=5., seed=0, **kwargs):
        self.name = name
        if name == "forrester":
            self.true_func = forrester_function
            self.boundaries = torch.tensor([[0., 1.]])
            self.alpha_out = 3.
        elif name == "branin":
            l = kwargs.get("l", .8)
            self.true_func = lambda x: branin_function(x, l)
            self.boundaries = torch.tensor([[-5., 10.], [0., 15.]])
            self.alpha_out = 20.
        else:
            print("WARNING: Unknown name, you should consider adding it to the DataGenerator class")
            assert true_func is not None, "You should provide a true function"
            assert boundaries is not None, "You should provide the boundaries of the function"
            self.true_func = true_func
            self.boundaries = boundaries
            self.alpha_out = alpha_out
        
        self.ndim = self.boundaries.size(0)
        self.prob_outlier = prob_outlier
        self.alpha_reg = alpha_reg
        
        self.reg_noise = torch.distributions.Normal(0, .15)
        self.out_noise = torch.distributions.Normal(2, 1.)
        self.seed = seed
        
        self.x = torch.concatenate([val.reshape(-1, 1) for val in torch.meshgrid(*[torch.linspace(*b, 100) for b in self.boundaries], indexing="ij")], axis=-1)
        # self.x = self.rescale(torch.from_numpy(lhsmdu.sample(self.ndim, (self.ndim*10))).T)
        if self.ndim == 1:
            self.x = self.x.squeeze(1)
        self.y = self.true_func(self.x)
        if seed is not None:
            torch.manual_seed(seed)
        lhsmdu.setRandomSeed(seed)
    
    def true_func_out(self, x):
        y = self.true_func(x)
        mask = torch.rand(x.size(0)) <= self.prob_outlier
        y += self.alpha_out * mask.float() * self.out_noise.sample(y.size()) + (1 - mask.float()) * self.reg_noise.sample(y.size()) * self.alpha_reg
        return y, mask
    
    def rescale(self, x):
        for i, bound in enumerate(self.boundaries):
            xmin, xmax = bound
            x[:, i] = xmin + (xmax - xmin) * x[:, i]
        return x
    
    def initial_lhs(self, n):
        x = self.rescale(torch.from_numpy(lhsmdu.sample(self.ndim, n, randomSeed=self.seed)).T)
        if self.ndim == 1:
            x = x.squeeze(1)
        y, outliers = self.true_func_out(x)
        return x, y, outliers
    
    def generate(self, x):
        y, outliers = self.true_func_out(x)
        return y, outliers
    
    

def generate_1d_data(n, n_out, alpha=1., xmin=0., xmax=.7,  seed=0):
    torch.manual_seed(seed)

    _x = torch.empty(n).normal_(0.35, 0.15)
    x_reg = _x[(_x > 0.0) & (_x < 0.7)]
    y_reg = forrester_function(x_reg) + torch.empty_like(x_reg).normal_(0, 0.15)
    n = x_reg.size(0)

    x_out = torch.empty(n_out).uniform_(xmin, xmax)
    y_out = forrester_function(x_out) + alpha * torch.empty_like(x_out).normal_(2, 1.).abs() * (torch.sign(torch.rand(n_out) - 0.5))

    # Create a tensor that indicates the source of each element
    source = torch.cat([torch.zeros(x_reg.size(0)), torch.ones(x_out.size(0))]).bool()

    return torch.cat([x_reg, x_out]), torch.cat([y_reg, y_out]), source


def generate_2d_data(n_points, n_outliers, alpha=20., step=.5, l=.8, seed=0):
    assert n_points + n_outliers < int(15 / step)**2, "Too many points for the step size of the grid"
    torch.manual_seed(seed)
    
    x = torch.arange(-5, 10 + step, step)
    y = torch.arange(0, 15 + step, step)
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    x_test = torch.stack((xv.flatten(), yv.flatten())).T
    y_test = branin_function(x_test, l)
    # print(x_test.shape, y_test.shape)
    
    perms = torch.randperm(x_test.shape[0])
    ix_reg = perms[:n_points]
    ix_out = perms[n_points:n_points+n_outliers]

    x_reg = x_test[ix_reg]
    y_reg = y_test[ix_reg] + torch.empty(n_points).normal_(0, 0.15) * 10.

    x_out = x_test[ix_out]
    y_out = y_test[ix_out] + alpha * torch.empty(n_outliers).normal_(2, .5).abs() * torch.sign(torch.rand(n_outliers) - 0.5)

    outliers = torch.cat([torch.zeros(n_points), torch.ones(n_outliers)]).bool()

    x_train = torch.cat([x_reg, x_out])
    y_train = torch.cat([y_reg, y_out])
    
    return xv, yv, x_test, y_test, x_train, y_train, outliers

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    name = "forrester"
    if name == "forrester":
        gen_1d = DataGeneratorOutliers(name, prob_outlier=.2, seed=None)
        x, y, outliers = gen_1d.initial_lhs(10)
        new_x = torch.tensor([0.5])
        new_y, mask = gen_1d.generate(new_x)
        plt.plot(gen_1d.x, gen_1d.y)
        plt.scatter(x[~outliers], y[~outliers], c="orange")
        plt.scatter(x[outliers], y[outliers], c="red")
        plt.scatter(new_x, new_y, c="green")
        plt.show()
    elif name == "branin":
        gen_2d = DataGeneratorOutliers(name, prob_outlier=.5, seed=None)
        x, y, outliers = gen_2d.initial_lhs(10)
        new_x = torch.tensor([[5, 10.]])
        new_y, mask = gen_2d.generate(new_x)
        plt.scatter(gen_2d.x[:, 0], gen_2d.x[:, 1], c=gen_2d.y)
        plt.scatter(x[:, 0][~outliers], x[:, 1][~outliers], c="orange")
        plt.scatter(x[:, 0][outliers], x[:, 1][outliers], c="red")
        plt.scatter(new_x[:, 0], new_x[:, 1], c="green")
    plt.show()
    print("done")