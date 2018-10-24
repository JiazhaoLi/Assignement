import numpy as np
import matplotlib.pyplot as plt


# feel free to read the two examples below, try to understand them
# in this problem, we require you to generate contour plots

# generate contour plot for function z = x^2 + 2*y^2
def plot_contour():

    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    plt.axis("square")
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    plt.show()


# generate heat plot (image-like) for function z = x^2 + 2*y^2
def plot_heat():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none', extent=[-3.0, 3.0, -3.0, 3.0],cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()



# This function receives the parameters of a multivariate Gaussian distribution
# over variables x_1, x_2 .... x_n as input and compute the marginal
#
def marginal_for_guassian(sigma,mu,given_indices):
    # given selected indices, compute marginal distribution for them
    pass


def conditional_for_gaussian(sigma,mu,given_indices,given_values):
    # given some indices that have fixed value, compute the conditional distribution
    # for rest indices
    pass


test_sigma_1 = np.array(
    [[1.0, 0.5],
     [0.5, 1.0]]
)

test_mu_1 = np.array(
    [0.0, 0.0]
)

test_sigma_2 = np.array(
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 1.5],
     [0.0, 0.0, 2.0, 0.0],
     [0.0, 1.5, 0.0, 4.0]]
)

test_mu_2 = np.array(
    [0.5, 0.0, -0.5, 0.0]
)

indices_1 = np.array([0])

indices_2 = np.array([1,2])
values_2 = np.array([0.1,-0.2])

plot_contour()
plot_heat()

marginal_for_guassian(test_sigma_1, test_mu_1, indices_1)
conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)