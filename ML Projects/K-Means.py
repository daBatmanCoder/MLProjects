import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(k):
        m = np.random.randint(0, X.shape[0])
        centroids.append(X[m])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 

def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for centroid in centroids:
        dis = np.sum(np.abs(X-centroid)**(p), axis=1)
        distances.append(dis**(1/p))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return np.array(distances)

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis = 0)
        new_centroids = np.array([np.mean(X[np.where(classes == j)], axis=0) for j in range(k)])
        if np.array_equal(new_centroids, centroids): 
            print(i)
            break
        centroids = new_centroids
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, np.asarray(classes)

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    centroids.append(X[np.random.choice(X.shape[0], 1, replace=False)])
    distances = lp_distance(X, centroids, 2)
    
    for i in range(k-1):
        distances = lp_distance(X, centroids, 2)
        minDis = []
        minDis = np.min(distances, axis=0)
        probs = minDis / sum(minDis)
        new_centroid = X[np.random.choice(X.shape[0], p=probs)]
        centroids.append(new_centroid)
        
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis = 0)
        new_centroids = np.array([X[np.where(classes == j)].mean(axis=0) for j in range(k)])
        if np.array_equal(new_centroids, centroids): 
            print(i)
            break
        centroids = new_centroids
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, np.array(classes)
