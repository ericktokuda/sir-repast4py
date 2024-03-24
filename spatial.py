from scipy.spatial import KDTree
def compute_d_far_points(points, d):
    kdtree = KDTree(points)
    d_distant_indices = kdtree.query_pairs(d)
    d_distant_pairs = [(i, j) for i, j in d_distant_indices]
    return d_distant_pairs

