#!/usr/bin/python
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

##############################################################################
# Generate sample data
#np.random.seed(0)

#centers = [[1, 1], [-1, -1], [1, -1]]
#n_clusters = len(centers)
#X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

#print X

with open('trycry.csv', 'rb') as f:

    X = np.array([])    
    for line in f:
        x, y, z, a, b, c, e, g, h, i, j, k= line.strip().split(",")
        x, y, z, a, b, c, e, g, h, i, j, k= float(x),float(y),float(z),float(a),float(b),float(c),float(e),float(g),float(h),float(i),float(j),float(k)
        X = np.append(X,[x,y,z,a,b,c,e,g,h,i,j,k])
#    print X
    
    Y = X.reshape(150,12)
#    print type(Y)

##############################################################################
# Compute clustering with Means

k_means = KMeans(init='k-means++', n_clusters=5, n_init=10)
t0 = time.time()
k_means.fit(Y)
t_batch = time.time() - t0
k_means_labels = k_means.labels_

k_means_cluster_centers = k_means.cluster_centers_
k_means_inertia = k_means.inertia_

c0 = k_means_cluster_centers[1]-k_means_cluster_centers[0]
c0_dist = math.sqrt((math.pow(c0[1],2)+math.pow(c0[0],2)))
c1 = k_means_cluster_centers[2]-k_means_cluster_centers[0]
c1_dist = math.sqrt((math.pow(c1[1],2)+math.pow(c1[0],2)))
c2 = k_means_cluster_centers[2]-k_means_cluster_centers[1]
c2_dist = math.sqrt((math.pow(c2[1],2)+math.pow(c2[0],2)))
center_sum = c0_dist + c1_dist + c2_dist
k_proper = k_means_inertia/center_sum
#print k_means_cluster_centers,
#print k_means_inertia,center_sum,k_proper
print k_means_labels
print k_proper

n_clusters= 5

##############################################################################
# Plot result

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
colors = ['#4EACC5', '#FF9C34', '#4E9A06','#FF0000','#FFE153']


#KMeans
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(Y[my_members, 0], Y[my_members,1],'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))


plt.show()
