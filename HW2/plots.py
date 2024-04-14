import matplotlib.pyplot as plt, numpy as np

data = np.loadtxt('output.dat')
centroids = np.loadtxt('centroids.dat')  # Load centroid data

cluster_ids = data[:, 1].astype(int)
points = data[:, 2:]

dimensions = points.shape[1]

clusters = np.unique(cluster_ids)
colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

if dimensions == 2:
    # 2D Plot
    for cluster, color in zip(clusters, colors):
        cluster_points = points[cluster_ids == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {cluster}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')  # Plot centroids
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
elif dimensions == 3:
    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cluster, color in zip(clusters, colors):
        cluster_points = points[cluster_ids == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=color, label=f'Cluster {cluster}')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black', marker='x', label='Centroids')  # Plot centroids
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
else:
    print(f"Data with {dimensions} dimensions cannot be plotted.")
    exit()

plt.legend(bbox_to_anchor=(1, .75), loc='upper left')

plt.savefig('figures/hepta.pdf', transparent=True, bbox_inches='tight')