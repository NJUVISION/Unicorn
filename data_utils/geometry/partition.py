
import numpy as np

def kdtree_partition(points, max_num, n_parts=None):
    parts = []
    if n_parts is not None: max_num = len(points)/n_parts + 2
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data)/2)]
        root = KD_node(point)
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        return root
    init_root = KD_node(None)
    root = createKDTree(init_root, points)

    return parts

def ravel_multi_index(data):
    # 3D -> 1D by sum each dimension
    data = data.astype('int64')
    step = data.max() + 1     
    data_1d = sum([data[:,i]*(step**i) for i in range(data.shape[-1])])
    
    return data_1d