########## PCA  ############
    
    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

X = pd.DataFrame(datasets.load_iris().data)


def my_pca(data_matrix, k):
    """
    Return matrix reduced to k dimensions with PCA.
    """
    cov_matrix = np.cov(data_matrix.transpose())
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues.sort()
    # sorts the eigenvalues in ascending order
    decending_eigenvalues = eigenvalues[-k:][::-1]
    # choose the highest k values and change the order to decending
    
    evalues, evectors = np.linalg.eig(cov_matrix)
    
    index_list = []
    for i in decending_eigenvalues:
        indexes = np.where(i == evalues)[0][0]
        index_list.append(indexes)
    
    
    evector_list = []
    for i in index_list:
        evector_list.append(evectors[i])
    
    evector_array = np.array(evector_list)
    
    reduced_matrix = np.dot(data_matrix, evector_array.transpose())
    
    return pd.DataFrame(reduced_matrix)

def my_pca_plot(low_dim_data):
    """
    Plot the data
    """
    if len(low_dim_data.columns) == 2: 
        x = low_dim_data[0]
        y = low_dim_data[1]
        plt.scatter(x, y, alpha=0.5)
        plt.title('PCA K = 2')
        plt.xlabel('x Data')
        plt.ylabel('y Data')
        
    
    if len(low_dim_data.columns) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(low_dim_data[0], low_dim_data[1], 
                   low_dim_data[2], c='skyblue', s=60)
        ax.view_init(30, 185)
        ax.set_xlabel("x Data")
        ax.set_ylabel("y Data")
        ax.set_zlabel("z Data")
        ax.set_title("PCA K = 3")

    
    plt.show() #plot 
    
    
PCA_K_3 = my_pca(X, 3)
PCA_K_2 = my_pca(X, 2)

my_pca_plot(PCA_K_2)
my_pca_plot(PCA_K_3)
