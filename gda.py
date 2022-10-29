import numpy as np
import matplotlib.pyplot as plt
import util
import math


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    print(">>Fitting the model...")
    clf.fit(x_train, y_train)
    print("FIT DONE")
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    y_prediction = clf.predict(x_valid)
    assert(y_prediction.shape == y_valid.shape), "The predicted vector y is of the wrong shape ; try again!"
    np.savetxt(save_path,y_prediction)
    if train_path == 'data/data_kckm_err_train.csv':
        util.plot_3d(x_valid,y_valid,clf.theta,save_path[:-3]+'png')
    else:
        util.plot(x_valid,y_valid,clf.theta,save_path[:-3]+'png')
    print(clf.theta)
    print("VALIDATION PLOT DONE")
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples =  x.shape[0]
        dim = x.shape[1]
        self.theta = np.zeros(dim+1)

        # Compute phi, mu_0, mu_1, sigma
        phi = np.sum(y == 1) / n_examples
        mu_0 = np.sum(x[y == 0], axis=0) / (np.sum(y == 0))
        mu_1 = np.sum(x[y == 1], axis=0) / np.sum(y == 1)
        sigma = (np.dot((x[y == 0] - mu_0).transpose(),x[y == 0] - mu_0) + np.dot((x[y == 1] - mu_1).transpose(),x[y == 1] - mu_1))/n_examples
        # Compute theta
        self.theta[0] = 0.5 * np.dot(np.dot(mu_0 + mu_1,np.linalg.inv(sigma)),mu_0 - mu_1)- np.log((1 - phi) / phi)
        self.theta[1:] = np.dot(np.linalg.inv(sigma),mu_1 - mu_0 )
        # # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return  1 / (1 + np.exp(-np.dot(x,self.theta)))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='data/data_kckm_train.csv',valid_path='data/data_kckm_valid.csv',save_path='gda_kckm.txt')
    main(train_path='data/data_err_train.csv',valid_path='data/data_err_valid.csv',save_path='gda_err.txt')
    main(train_path='data/data_kckm_err_train.csv',valid_path='data/data_kckm_err_valid.csv',save_path='gda_kckm_err.txt')

