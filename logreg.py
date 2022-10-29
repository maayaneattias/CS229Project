import numpy as np
import matplotlib.pyplot as plt
import util
import math

#TO DO:
#proportion valid train:0.75 change?
#log reg vs gda: kcat kmean
#comment on number of iterations and decision boudary comment on singular matrix
#log reg gda: error_cat error_mean
#log reg gda: kcat kmean sum of error in 3d

#different gda and log reg for each set ; difference datasets


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    #Fitting model
    clf = LogisticRegression(max_iter=100000)
    print(">>Fitting the model...")
    clf.fit(x_train, y_train)
    print("FIT DONE")
    #Prediction on validation data
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
    
class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])
        i = 0
        while True:
            i+=1
            y_pred = self.predict(x)
            grad = ((y_pred - y) * x.T).mean(axis=1)
            hess = ((y_pred * (1 - y_pred)) * x.T) @ x / x.shape[1]
            if np.linalg.det(hess.T)==0:
                print("The algorithm converged in {} iterations".format(i))
                return self
            diff = grad @ np.linalg.inv(hess.T)
            self.theta = self.theta - diff
            if np.abs(diff).sum() < self.eps or i>=self.max_iter:
                print("The algorithm converged in {} iterations".format(i))
                return self
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return  1 / (1 + np.exp(-np.dot(x,self.theta)))

        # # *** END CODE HERE ***
    

if __name__ == '__main__':
    main(train_path='data/data_kckm_train.csv',valid_path='data/data_kckm_valid.csv',save_path='logreg_kckm.txt')
    main(train_path='data/data_err_train.csv',valid_path='data/data_err_valid.csv',save_path='logreg_err.txt')
    main(train_path='data/data_kckm_err_train.csv',valid_path='data/data_kckm_err_valid.csv',save_path='logreg_kckm_err.txt')

