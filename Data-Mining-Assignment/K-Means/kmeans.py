import matplotlib.pyplot as plt
# import imageio as io
from sklearn import cluster

from skimage import io
image = io.imread('monalisa.jpg', as_gray=True)
# image = io.imread("monalisa.jpg")
plt.figure(figsize = (8,8))
plt.imshow(image)
plt.show()


def kmeans_on_image(cluster_number,image_l,  x, y):

    kmeans_cluster = cluster.KMeans(n_clusters=cluster_number)
    kmeans_cluster.fit(image_l)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    return cluster_centers[cluster_labels].reshape(x, y)


print(image.shape)
x, y = image.shape
z=1
image_2d = image.reshape(x*y, z)
print(image_2d.shape)


for i in range(2,10):
    plt.figure(figsize = (8,8))
    print("Running k means with k=", i)
    plt.imshow(kmeans_on_image(i, image_2d, x, y))
    plt.show()