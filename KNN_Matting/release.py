import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2


def knn_matting(image, trimap, my_lambda=100):
    [h, w, c] = image.shape
    image, trimap = image / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)

    all_constraints = foreground + background

    print('Finding nearest neighbors')
    a, b = np.unravel_index(np.arange(h*w), (h, w))
    feature_vec = np.append(np.transpose(image.reshape(h*w,c)), [ a, b]/np.sqrt(h*h + w*w), axis=0).T
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    print('Computing sparse A')
    row_inds = np.repeat(np.arange(h*w), 10)
    col_inds = knns.reshape(h*w*10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(c+2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(h*w, h*w))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script-A
    D = scipy.sparse.diags(np.ravel(all_constraints[:,:]))
    v = np.ravel(foreground[:,:])
    c = 2*my_lambda*np.transpose(v)
    H = 2*(L + my_lambda*D)

    print('Solving linear system for alpha')
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(h, w)
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(h, w)
    return alpha


if __name__ == '__main__':

    image = cv2.imread('./image/gandalf.png')
    trimap = cv2.imread('./trimap/gandalf.png', cv2.IMREAD_GRAYSCALE)

    alpha = knn_matting(image, trimap)
    alpha = alpha[:, :, np.newaxis]

    plt.title('Alpha Matte')
    plt.imshow(alpha, cmap='gray')
    plt.show()

    """
    result = []
    cv2.imwrite('./result/bear.png', result)
    """
    
