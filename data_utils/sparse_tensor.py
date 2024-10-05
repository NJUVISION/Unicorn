
import torch
import numpy as np
import MinkowskiEngine as ME


def array2vector(array, step=None):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().clone(), step.long().clone()
    if array.min()<0:
        min_value = array.min()
        array = array - min_value
        step = step - min_value
        
    assert array.min()>=0 and array.max()-array.min()<step
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    data = data.clone()
    ground_truth = ground_truth.clone()
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    # positive value
    min_value =  torch.min(data.min(), ground_truth.min())
    if min_value < 0:
        data[:,1:] -= min_value
        ground_truth[:,1:] -= min_value
    #
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask


def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    sparse_tensor = ME.SparseTensor(features=features, 
                                coordinates=coordinates,
                                tensor_stride=tensor_stride,
                                device=device)

    return sparse_tensor


def sort_sparse_tensor(sparse_tensor, target=None):
    """ Sort points in sparse tensor according to their coordinates or the coords of target
    """
    if target is not None and (sparse_tensor.C==target.C).all():
        return ME.SparseTensor(features=sparse_tensor.F, 
                            coordinate_map_key=target.coordinate_map_key, 
                            coordinate_manager=target.coordinate_manager, 
                            device=target.device)

    # positive value
    coords = sparse_tensor.C.clone()
    min_value =  coords.min()
    if min_value < 0: coords[:,1:] -= min_value
    # sort
    indices = torch.argsort(array2vector(coords, coords.max()+1))
    out_coords = sparse_tensor.C[indices]
    out_feats = sparse_tensor.F.cpu()[indices]
    out = create_new_sparse_tensor(coordinates=out_coords, features=out_feats, 
                                tensor_stride=sparse_tensor.tensor_stride, 
                                dimension=sparse_tensor.D, device=sparse_tensor.device)
    if target is not None:
        # positive value
        target_coords = target.C.clone()
        min_value =  target_coords.min()
        if min_value < 0: target_coords[:,1:] -= min_value
        # sort
        target_indices = torch.argsort(array2vector(target_coords, target_coords.max()+1))
        inverse_indices = target_indices.sort()[1]
        assert (out_coords[inverse_indices]==target.C).all()
        out = ME.SparseTensor(features=out_feats[inverse_indices], 
                            coordinate_map_key=target.coordinate_map_key, 
                            coordinate_manager=target.coordinate_manager, 
                            device=target.device)

    return out


from pytorch3d.ops.knn import knn_points, knn_gather

def knn_fn(coords0, coords1, feats0, knn):
    coords0 = coords0.unsqueeze(0)# [1,n0,3]
    coords1 = coords1.unsqueeze(0)# [1,n1,3]
    dists, knn_idxs, _ = knn_points(coords1, coords0, K=knn, 
                                    return_nn=False, return_sorted=True)
    assert (dists.squeeze(0)-((coords1.squeeze(0).unsqueeze(1).expand(-1,knn,-1) - \
        coords0.squeeze(0)[knn_idxs.squeeze(0)])**2).sum(dim=-1)).abs().min()<1e-3
    assert knn_idxs.shape[1]==coords1.shape[1]
    # relative positions
    knn_coords = knn_gather(coords0, knn_idxs)# [1,n1,k,3]
    knn_coords = knn_coords - coords1.unsqueeze(-2)# [1,n1,k,3]
    knn_coords = knn_coords.squeeze(0)# [n1,k,3]
    # # normalized inverse distance
    dists = dists.squeeze(0) #[n1, k]
    # knn feature
    knn_feats = knn_gather(feats0.unsqueeze(0), knn_idxs).squeeze(0)# [n1,k,d]
    assert (knn_feats==feats0[knn_idxs.squeeze(0)]).all()
    
    return knn_coords, knn_feats, dists, knn_idxs


def knn_interpolation_fn(knn_feats, knn_dists, min_norm=0.01):
    """dists: [n1,k];   feats: [n1,k,d]
        weighted sum by normalized inverse distance.
    """
    knn_dists = knn_dists.clamp(min=1e-8)# [n1, k]
    weights = 1.0/knn_dists# [n1, k]
    norm = torch.sum(weights, dim=-1, keepdim=True).clamp(min_norm)# [n1, 1]
    weights = weights/norm# [n1, k]
    feats = torch.einsum('nk,nkd->nd', weights, knn_feats)
    if True:
        feats_tp = (knn_feats * weights.unsqueeze(-1)).sum(dim=1)
        error = (feats - feats_tp).detach().cpu().abs().max()
        assert error<1e-3
    
    return feats
