import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

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

# Set up K-Means clustering with a fixed start and stop at 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

# Plot the data
sns.set_style("darkgrid")
plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, cmap=plt.get_cmap("winter"))

# Save the axes limits of the current figure
x_axis = plt.gca().get_xlim()
y_axis = plt.gca().get_ylim()

# Draw cluster boundaries and centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='x')
vor = Voronoi(centers)
voronoi_plot_2d(vor, ax=plt.gca(), show_points=False, show_vertices=False)

# Resize figure as needed
plt.gca().set_xlim(x_axis)
plt.gca().set_ylim(y_axis)

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
