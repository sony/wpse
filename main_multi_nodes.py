# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------

import os
import hydra
import torch
import torch.multiprocessing as mp

import main

def wrapper(local_rank, node_rank, n_gpu, args):
    args.gpu = local_rank
    args.rank = node_rank * n_gpu + local_rank
    main.main(args)

@hydra.main(version_base=None, config_path="configs")
def spawn_nodes(args):
    # ddp setup
    args.dist_backend = "nccl"
    args.dist_url = f"tcp://{args.hostname}:{args.port_num}"

    node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])  # Process number in MPI
    n_node = int(os.environ["OMPI_COMM_WORLD_SIZE"])  # The number of nodes
    n_gpu = torch.cuda.device_count()  # The number of gpus per node
    args.world_size = n_gpu * n_node

    # spawn a process
    mp.spawn(wrapper, nprocs=n_gpu, args=(node_rank, n_gpu, args))


if __name__ == "__main__":
    spawn_nodes()
