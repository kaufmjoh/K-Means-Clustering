import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(42)

# Toy problem with 3 clusters for us to verify k-means is working well
def toyProblem():
  # Generate a dataset with 3 cluster
  X = np.random.randn(150,2)*1.5
  X[:50,:] += np.array([1,4])
  X[50:100,:] += np.array([15,-2])
  X[100:,:] += np.array([5,-2])


  #print(X)

  # Randomize the seed
  np.random.seed()

  # Apply kMeans with visualization on
  k = 3
  max_iters=20
  #centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=True)
  centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)

  #print(X[0:2, ])

  #raise Exception('Stop\n')
  print("Loading Clustering Plot 1")
  plotClustering(centroids, assignments, X, title="Final Clustering")

  #raise Exception('Stop\n')
  # Print a plot of the SSE over training
  print("Loading SSE Chart")
  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.show()


  #raise Exception('Stop\n')
  #############################
  # Q5 Randomness in Clustering
  #############################
  k = 5
  max_iters = 20

  SSE_rand = []
  SSE_rand = np.zeros(50)
  # Run the clustering with k=5 and max_iters=20 fifty times and
  # store the final sum-of-squared-error for each run in the list SSE_rand.
  #raise Exception('Student error: You haven\'t implemented the randomness experiment for Q5.')

  for i in range(50):
    print("Performing 5-means Clustering: ", i)
    centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
    #print(SSE)
    SSE_rand[i] = SSE[max_iters-1]
    #plotClustering(centroids, assignments, X, title="Final Clustering")



  # Plot error distribution
  plt.figure(figsize=(8,8))
  plt.hist(SSE_rand, bins=20)
  plt.xlabel("SSE")
  plt.ylabel("# Runs")
  plt.show()

  ########################
  # Q6 Error vs. K
  ########################

  SSE_vs_k = []
  SSE_vs_k = np.zeros(150)
  # Run the clustering max_iters=20 for k in the range 1 to 150 and
  # store the final sum-of-squared-error for each run in the list SSE_vs_k.
  #raise Exception('Student error: You haven\'t implemented Q6.')
  for i in range(150):
      print("Performing ", (i+1), " means clustering")
      centroids, assignments, SSE = kMeansClustering(X, k=(i+1), max_iters=max_iters, visualize=False)
      SSE_vs_k[i] = SSE[max_iters-1]
      #plotClustering(centroids, assignments, X, title="Final Clustering")


  # Plot how SSE changes as k increases
  plt.figure(figsize=(16,8))
  plt.plot(SSE_vs_k, marker="o")
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.show()


def imageProblem():
  print("Beginning Image Problem")

  np.random.seed()
  # Load the images and our pre-computed HOG features
  data = np.load("img.npy")
  img_feats = np.load("hog.npy")


  # Perform k-means clustering
  print("Performing Clustering Algorithm")
  k=3
  centroids, assignments, SSE = kMeansClustering(img_feats, k, 30, min_size=0)


  #print(SSE, k)
  print("Clustering Complete")

  # Visualize Clusters
  for c in range(len(centroids)):
    # Get images in this cluster
    members = np.where(assignments==c)[0].astype(np.int)
    imgs = data[np.random.choice(members,min(50, len(members)), replace=False),:,:]

    # Build plot with 50 samples
    print("Cluster "+str(c) + " ["+str(len(members))+"]")
    _, axs = plt.subplots(5, 10, figsize=(16, 8))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img,plt.cm.gray)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # Fill out plot with whitespace if there arent 50 in the cluster
    for i in range(len(imgs), 50):
      axs[i].axes.xaxis.set_visible(False)
      axs[i].axes.yaxis.set_visible(False)
    plt.show()



##########################################################
# initializeCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   k --  integer number of clusters to make
#
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
##########################################################

def initalizeCentroids(dataset, k):
  #print("debug1\n")

  indices = random.sample(range(dataset.shape[0]), k)
  #print(indices)


  centroids = np.zeros((k, dataset.shape[1]))

  for i in range(k):
    centroids[i] = dataset[indices[i]]
  #print(centroids)

  #raise Exception('Student error: You haven\'t implemented initializeCentroids yet.')
  return centroids

##########################################################
# computeAssignments
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#
# Outputs:
#   assignments -- n x 1 matrix of indexes where the i'th
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
##########################################################

def computeAssignments(dataset, centroids):
  #print("debug2\n")

  point_distance = np.copy(dataset)
  point_distance = np.repeat(point_distance[:,:,np.newaxis], centroids.shape[0], axis=2)

  #print(point_distance.shape)

  point_distance -= np.transpose(centroids)

  distances = np.linalg.norm(point_distance, axis = 1)

  #print(distances.shape)

  #distances = np.zeros((dataset.shape[0], centroids.shape[0]))
  #point_distance = np.zeros((dataset.shape[0], dataset.shape[1]))
  #assignments = np.zeros(dataset.shape[0])

  #for i in range(centroids.shape[0]):
  #  point_distance = dataset-centroids[i] ##find the distance between coordinates
  #  distances[:,i] = np.linalg.norm(point_distance, axis=1) #l2 distance

  assignments = np.argsort(distances, axis=1)[:,0] #take the shortest distance

  #print(assignments)

  #raise Exception('Student error: You haven\'t implemented computeAssignments yet.')
  return assignments



##########################################################
# updateCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j after being updated
#                 as the mean of assigned points
#   counts -- k x 1 matrix where the j'th entry is the number
#             points assigned to cluster j
##########################################################

def updateCentroids(dataset, centroids, assignments):

  #print("debug3\n")

  #print(centroids)

  summation = np.zeros((centroids.shape[0], dataset.shape[1]))
  curr_cluster = -1

  counts = np.zeros((centroids.shape[0],1))

  for i in range(dataset.shape[0]):
      curr_cluster = assignments[i]
      summation[curr_cluster] += dataset[i]
      counts[curr_cluster] += 1

  if counts.size - np.count_nonzero(counts) == 0:
    centroids = summation / counts
  else:
    for i in range(centroids.shape[0]):
      if counts[i] != 0:
        centroids[i] = summation[i]/counts[i]

  #print(centroids)

  #raise Exception('Student error: You haven\'t implemented updateCentroids yet.')
  return centroids, counts


##########################################################
# calculateSSE
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   sse -- the sum of squared error of the clustering
##########################################################

def calculateSSE(dataset, centroids, assignments):
  #print("debug4\n")


  sse = 0
  coordinate_d = np.copy(dataset)

  for i in range(dataset.shape[0]):
      curr_cluster = assignments[i]
      coordinate_d[i] -= centroids[curr_cluster]


  line_d = np.linalg.norm(coordinate_d, axis = 1)

  sse = np.sum(line_d)

  #raise Exception('Student error: You haven\'t implemented calculateSSE yet.')
  return sse


########################################
# Instructor Code: Don't need to modify
# beyond this point but should read it
########################################

def kMeansClustering(dataset, k, max_iters=10, min_size=0, visualize=False):

  # Initialize centroids
  centroids = initalizeCentroids(dataset, k)

  #print(centroids)

  # Keep track of sum of squared error for plotting later
  SSE = []

  # Main loop for clustering
  for i in range(max_iters):

    #print(centroids)
    # Update Assignments Step
    assignments = computeAssignments(dataset, centroids)

    # Update Centroids Step
    centroids, counts = updateCentroids(dataset, centroids, assignments)

    # Re-initalize any cluster with fewer then min_size points
    for c in range(k):
      if counts[c] <= min_size:
        centroids[c] = initalizeCentroids(dataset, 1)

    if visualize:
      plotClustering(centroids, assignments, dataset, "Iteration "+str(i))
    SSE.append(calculateSSE(dataset,centroids,assignments))

    # Get final assignments
    assignments = computeAssignments(dataset, centroids)

  return centroids, assignments, SSE

def plotClustering(centroids, assignments, dataset, title=None):
  plt.figure(figsize=(8,8))
  plt.scatter(dataset[:,0], dataset[:,1], c=assignments, edgecolors="k", alpha=0.5)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=5, edgecolors="k", s=250)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=2, edgecolors="w", s=200)
  if title is not None:
    plt.title(title)
  plt.show()


if __name__=="__main__":
  toyProblem()
  imageProblem()
