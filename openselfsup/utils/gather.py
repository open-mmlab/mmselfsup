import numpy as np

import torch
import torch.distributed as dist


def gather_tensors(input_array):
    world_size = dist.get_world_size()
    ## gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [
        torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)
    ]
    dist.all_gather(all_shape, shape_tensor)
    ## compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    ## padding tensors and gather them
    output_tensors = [
        torch.Tensor(max_count).cuda() for i in range(world_size)
    ]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)
    ## unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [
        x[:all_count[i]].reshape(all_shape[i])
        for i, x in enumerate(padded_output)
    ]
    return output


def gather_tensors_batch(input_array, part_size=100, ret_rank=-1):
    # batch-wize gathering to avoid CUDA out of memory
    rank = dist.get_rank()
    all_features = []
    part_num = input_array.shape[0] // part_size + 1 if input_array.shape[
        0] % part_size != 0 else input_array.shape[0] // part_size
    for i in range(part_num):
        part_feat = input_array[i *
                                part_size:min((i + 1) *
                                              part_size, input_array.shape[0]),
                                ...]
        assert part_feat.shape[
            0] > 0, "rank: {}, length of part features should > 0".format(rank)
        #print("rank: {}, gather part: {}/{}, length: {}".format(rank, i, part_num, len(part_feat)))
        gather_part_feat = gather_tensors(part_feat)
        all_features.append(gather_part_feat)
    if ret_rank == -1:
        all_features = [
            np.concatenate([all_features[i][j] for i in range(part_num)],
                           axis=0) for j in range(len(all_features[0]))
        ]
        return all_features
    else:
        if rank == ret_rank:
            all_features = [
                np.concatenate([all_features[i][j] for i in range(part_num)],
                               axis=0) for j in range(len(all_features[0]))
            ]
            return all_features
        else:
            return None
