
import torch
import torch.nn.functional as F
import numpy as np


class PathIndex:

    def __init__(self, radius=5, default_size=None):

        self.radius = radius
        self.radius_floor = int(np.ceil(radius) - 1)

        self.path_list_by_length, self.path_dst = self.get_all_dir_paths(self.radius)
        self.path_dst2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.path_dst).transpose(1, 0), 0), -1).float()

        if default_size:
            self.default_path_indices, self.default_src_indices, self.default_dst_indices \
                = self.get_path_indices(default_size)

        return

    def get_all_dir_paths(self, max_radius=5):

        coord_indices_by_length = [[] for _ in range(max_radius * 4)]

        search_dirs = []

        for x in range(1, max_radius):
            search_dirs.append((0, x))

        for y in range(1, max_radius):
            for x in range(-max_radius + 1, max_radius):
                if x * x + y * y < max_radius ** 2:
                    search_dirs.append((y, x))

        for dir in search_dirs:

            length_sq = dir[0] ** 2 + dir[1] ** 2
            path_coords = []

            min_y, max_y = sorted((0, dir[0]))
            min_x, max_x = sorted((0, dir[1]))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):

                    dist_sq = (dir[0] * x - dir[1] * y) ** 2 / length_sq

                    if dist_sq < 1:
                        path_coords.append([y, x])

            path_coords.sort(key=lambda x: -abs(x[0]) - abs(x[1]))
            path_length = len(path_coords)

            coord_indices_by_length[path_length].append(path_coords)

        path_list_by_length = [np.asarray(v) for v in coord_indices_by_length if v]
        path_destinations = np.concatenate([p[:, 0] for p in path_list_by_length], axis=0)

        return path_list_by_length, path_destinations

    def get_path_indices(self, size):

        full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1]))

        cropped_height = size[0] - self.radius_floor
        cropped_width = size[1] - 2 * self.radius_floor

        paths_by_length_list = []

        for paths in self.path_list_by_length:

            path_indices_list = []
            for p in paths:

                coord_indices_list = []

                for dy, dx in p:
                    coord_indices = full_indices[dy:dy + cropped_height,
                                    self.radius_floor + dx:self.radius_floor + dx + cropped_width]
                    coord_indices = np.reshape(coord_indices, [-1])

                    coord_indices_list.append(coord_indices)

                path_indices_list.append(coord_indices_list)

            paths_by_length_list.append(np.array(path_indices_list))

        src_indices = np.reshape(full_indices[:cropped_height, self.radius_floor:self.radius_floor + cropped_width], -1)
        dest_indices = np.concatenate([p[:,0] for p in paths_by_length_list], axis=0)

        return paths_by_length_list, \
               src_indices, \
               dest_indices

    def to_displacement(self, x):
        height, width = x.size(2), x.size(3)

        cropped_height = height - self.radius_floor
        cropped_width = width - 2 * self.radius_floor

        feat_src = x[:, :, :cropped_height, self.radius_floor:self.radius_floor + cropped_width]

        feat_dest = [x[:, :, dy:dy + cropped_height, self.radius_floor + dx:self.radius_floor + dx + cropped_width]
                       for dy, dx in self.path_dst]
        feat_dest = torch.stack(feat_dest, 2)

        disp = torch.unsqueeze(feat_src, 2) - feat_dest
        disp = disp.view(disp.size(0), disp.size(1), disp.size(2), -1)

        return disp

    def to_displacement_loss(self, x):

        return torch.abs(x - self.path_dst2.cuda())


def edge_to_affinity(edge, paths_indices):

    aff_list = []
    edge = edge.view(edge.size(0), -1)

    for i in range(len(paths_indices)):
        if isinstance(paths_indices[i], np.ndarray):
            paths_indices[i] = torch.from_numpy(paths_indices[i])
        paths_indices[i] = paths_indices[i].cuda(non_blocking=True)

    for ind in paths_indices:
        ind_flat = ind.view(-1)
        dist = torch.index_select(edge, dim=-1, index=ind_flat)
        dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
        aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
        aff_list.append(aff)
    aff_cat = torch.cat(aff_list, dim=1)

    return aff_cat


def feature_to_affinity(x, ind_from, ind_to):

    x = x.view(x.size(0), x.size(1), -1)

    if isinstance(ind_from, np.ndarray):
        ind_from = torch.from_numpy(ind_from)
        ind_to = torch.from_numpy(ind_to)

    ind_from = torch.unsqueeze(ind_from, dim=0)

    ff = x[..., ind_from.cuda(non_blocking=True)]
    ft = x[..., ind_to.cuda(non_blocking=True)]

    aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

    return aff


def affinity_sparse2dense(affinity_sparse, ind_from, ind_to, n_vertices):

    ind_from = torch.from_numpy(ind_from)
    ind_to = torch.from_numpy(ind_to)

    affinity_sparse = affinity_sparse.view(-1).cpu()
    ind_from = ind_from.repeat(ind_to.size(0)).view(-1)
    ind_to = ind_to.view(-1)

    indices = torch.stack([ind_from, ind_to])
    indices_tp = torch.stack([ind_to, ind_from])

    indices_id = torch.stack([torch.arange(0, n_vertices).long(), torch.arange(0, n_vertices).long()])

    affinity_dense = torch.sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                       torch.cat([affinity_sparse, torch.ones([n_vertices]), affinity_sparse])).to_dense().cuda()

    return affinity_dense

def to_transition_matrix(affinity_dense, beta, times):
    scaled_affinity = torch.pow(affinity_dense, beta)

    trans_mat = scaled_affinity / torch.sum(scaled_affinity, dim=0, keepdim=True)
    for _ in range(times):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    return trans_mat

def propagate_to_edge(x, edge, radius=5, beta=10, exp_times=8):

    height, width = x.shape[-2:]

    hor_padded = width+radius*2
    ver_padded = height+radius

    path_index = PathIndex(radius=radius, default_size=(ver_padded, hor_padded))

    edge_padded = F.pad(edge, (radius, radius, 0, radius), mode='constant', value=1.0)
    sparse_aff = edge_to_affinity(torch.unsqueeze(edge_padded, 0),
                                               path_index.default_path_indices)

    dense_aff = affinity_sparse2dense(sparse_aff, path_index.default_src_indices,
                                      path_index.default_dst_indices, ver_padded*hor_padded)
    dense_aff = dense_aff.view(ver_padded, hor_padded, ver_padded, hor_padded)
    dense_aff = dense_aff[:-radius, radius:-radius, :-radius, radius:-radius]
    dense_aff = dense_aff.reshape(height * width, height * width)

    trans_mat = to_transition_matrix(dense_aff, beta=beta, times=exp_times)

    x = x.view(-1, height, width) * (1 - edge)


    rw = torch.matmul(x.view(-1, height * width), trans_mat)
    rw = rw.view(rw.size(0), 1, height, width)

    return rw