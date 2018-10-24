import numpy as np
import matplotlib.pyplot as plt


# For this problem, we use data generator instead of real dataset
def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)

    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys


def main():
    noise_scales = [0.05,0.2]

    # for example, choose the first kind of noise scale
    noise_scale = noise_scales[0]

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)


    # bandwidth parameters
    sigma_paras = [0.1,0.2,0.4,0.8,1.6]

main()




