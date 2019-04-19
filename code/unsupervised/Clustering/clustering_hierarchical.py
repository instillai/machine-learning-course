import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from scipy.spatial import ConvexHull

# This data set represents a toy manufacturer's product data
#
# The first value in the pair represents a toy:
#    0-2: Action Figures
#    3-5: Building Blocks
#    6-8: Cars
#
# The second value is the age group that buys the most of that toy:
#    0: 5 year-olds
#    1: 6 year-olds
#    2: 7 year-olds
#    3: 8 year-olds
#    4: 9 year-olds
#    5: 10 year-olds
x = np.array([[0,4], [1,3], [2,5], [3,2], [4,0], [5,1], [6,4], [7,5], [8,3]])

# Set up hierarchical clustering and stop at 3 clusters
num_clusters = 3
hierarchical = AgglomerativeClustering(n_clusters=num_clusters).fit(x)

# Plot the data
sns.set_style("darkgrid")
colors = plt.get_cmap("winter")
points = plt.scatter(x[:, 0], x[:, 1], c=hierarchical.labels_,
            cmap=colors)

# Draw in the cluster regions
regions = defaultdict(list)
# Split points based on cluster
for index, label in enumerate(hierarchical.labels_):
    regions[label].append(list(x[index]))

# If a cluster has more than 2 points, find the convex hull for the region
# Otherwise just draw a connecting line
for key in regions:
    cluster = np.array(regions[key])
    if len(cluster) > 2:
        hull = ConvexHull(cluster)
        vertices = hull.vertices
        vertices = np.append(vertices, hull.vertices[0])
        plt.plot(cluster[vertices, 0], cluster[vertices, 1],
                 color=points.to_rgba(key))
    else:
        np.append(cluster, cluster[0])
        x_region, y_region = zip(*cluster)
        plt.plot(x_region, y_region, color=points.to_rgba(key))

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
