import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
plt.rcParams['text.usetex'] = True

# Total number of samples
m = 150

# 75 samples for the red line, 74 samples for the blue line, and a single blue outlier.
X = np.array([(-1, i) for i in np.linspace(0.25, 1, 75)] + [(1, i) for i in np.linspace(1.05, 3, 74)] + [(-80, 1.01)])
Y = [0] * 75 + [1] * 74 + [1]

# Draw the dataset
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm.get_cmap("RdBu"))
plt.title("Dataset ($m=150$, input features $x_1$ and $x_2$)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig("dataset.png", dpi=300)

# Perform the hyper-parameter search
lms = np.logspace(-3, 7, 16)
train_acc = []
val_acc = []
for i, lm in enumerate(lms):

    # Define the Linear SVM model (equivalent to KernelSVM with linear kernel)
    svm = LinearSVC(C=1/(lm * m), max_iter=10000000)

    # Perform 5-fold validation
    cv_results = cross_validate(svm, X, Y, cv=5, return_train_score=True)
    train_acc.append(np.mean(cv_results['train_score']).item())
    val_acc.append(np.mean(cv_results['test_score']).item())

# Draw the results
plt.figure()
plt.xlabel("$\lambda$")
plt.xscale("log")
plt.ylabel("Accuracy")
plt.plot(lms, train_acc)
plt.plot(lms, val_acc)
plt.legend(["Train", "Validation"])
plt.title("SVM 5-Fold Hyper-parameter Search")
plt.savefig("SVM.png", dpi=300)

