import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)


# Data generation function
def generate_data(m, w):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    while(True):
        X = np.random.multivariate_normal(mean, cov, m)
        y = np.sign(X @ w)  # Generate labels using the linear separator
        # check that at there are at list two different lables in y
        if len(np.unique(y)) > 1:
            break
    return X, y

# Function to plot SVM decision boundary
def plot_svm(X, y, clf, f_w, m, C, save_dir=None):
    plt.figure(figsize=(6, 6))
    plt.title(f"SVM with m={m}, C={C}")

    # Plot the data points with true labels
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=50)

    x_vals = np.linspace(-3, 3, 100)
    y_vals_true = -(f_w[0] / f_w[1]) * x_vals  # True decision boundary
    plt.plot(x_vals, y_vals_true, 'k--', label='True f(x)')

    # Plot SVM decision boundary: w[0]*x + w[1]*y + b = 0 => y = -(w[0]*x + b) / w[1]
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
        b = clf.intercept_[0]
        y_vals_svm = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals_svm, 'b-', label='SVM decision boundary')

    plt.legend()
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('x2')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        plt.savefig(os.path.join(save_dir, f"svm_m{m}_C{C}.png"))
        plt.close()
    else:
        plt.show()


def generate_moons_data():
    X, y = make_moons(n_samples=200, noise=0.2)  # Generate moon-shaped data
    return train_test_split(X, y, test_size=0.2, random_state=42)

def generate_circles_data():
    X, y = make_circles(n_samples=200, noise=0.1)  # Generate circle-shaped data
    return train_test_split(X, y, test_size=0.2, random_state=42)

def generate_gaussians_data():
    mean1 = [-1, -1]
    mean2 = [1, 1]
    cov = [[0.5, 0.2], [0.2, 0.5]]

    X1 = np.random.multivariate_normal(mean1, cov, 100)
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    X = np.vstack((X1, X2))  # Combine two Gaussian distributions
    y = np.array([0]*100 + [1]*100)  # Labels for the two distributions
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_classifiers():
    classifiers = {
        "Linear-SVM (λ=5)": SVC(C=1/5, kernel='linear'),  # Linear SVM with regularization
        "Decision Tree (depth=7)": DecisionTreeClassifier(max_depth=7),  # Decision tree with depth limit
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)  # K-Nearest Neighbors with k=5
    }
    return classifiers

# Function to plot decision boundary with background color and test accuracy in title
def plot_decision_boundary(clf, X_train, y_train, X_test, y_test, title, save_path=None):
    h = 0.02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict class for each point in the mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Evaluate accuracy on the test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Create color maps: light for background, bold for points
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # Plot decision boundary by background coloring
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=50, label='Train')
    # Plot test points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, s=20, marker='x', label='Test')

    plt.title(f"{title}\nTest Accuracy: {acc:.2f}")
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')

    if save_path:
        plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_')}.png"))
        plt.close()
    else:
        plt.show()


def pratical_1_runner(save_path=None):
    m_vals = [5, 10, 20, 100]  # Different dataset sizes
    C_vals = [0.1, 1, 5, 10, 100]  # Different SVM regularization values
    w = np.array([-0.6, 0.4])  # True linear separator
    for m in m_vals:
        X, y = generate_data(m, w)
        for C in C_vals:
            clf = SVC(C=C, kernel='linear')  # Train SVM with linear kernel
            clf.fit(X, y)
            plot_svm(X, y, clf, w, m, C, save_path)

def practical_2_runner(save_path=None):
    datasets = {
    "Moons": generate_moons_data,
    "Circles": generate_circles_data,
    "Gaussians": generate_gaussians_data
    }

    classifiers = get_classifiers()

    # Run and plot for each dataset and each classifier
    for dataset_name, data_fn in datasets.items():
        X_train, X_test, y_train, y_test = data_fn()
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)  # Train classifier
            plot_title = f"{clf_name} on {dataset_name}"
            plot_decision_boundary(clf, X_train, y_train, X_test, y_test, plot_title, save_path)


if __name__ == "__main__":
    path = None
    pratical_1_runner(save_path=path)
    practical_2_runner(save_path=path) 
