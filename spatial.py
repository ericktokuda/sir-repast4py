from scipy.spatial import KDTree
def compute_d_far_points(points, d):
    kdtree = KDTree(points)
    d_distant_indices = kdtree.query_pairs(d)
    return d_distant_indices

