import atexit
import socket

import pytest
import torch

from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.dist_neighbor_sampler import (
    DistNeighborSampler,
    close_sampler,
)
from torch_geometric.distributed.partition import load_partition_info
from torch_geometric.distributed.rpc import init_rpc
from torch_geometric.sampler import EdgeSamplerInput, NeighborSampler
from torch_geometric.sampler.neighbor_sampler import edge_sample
from torch_geometric.testing import withPackage


def create_hetero_data(tmp_path: str, rank: int):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    # Other partition graph store:
    graph_store_other = LocalGraphStore.from_partition(tmp_path,
                                                       pid=int(not rank))
    feat_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)
    (
        meta,
        num_partitions,
        partition_idx,
        node_pb,
        edge_pb,
    ) = load_partition_info(tmp_path, rank)
    graph_store.partition_idx = partition_idx
    graph_store.num_partitions = num_partitions
    graph_store.node_pb = node_pb
    graph_store.edge_pb = edge_pb
    graph_store.meta = meta

    feat_store.partition_idx = partition_idx
    feat_store.num_partitions = num_partitions
    feat_store.node_feat_pb = node_pb
    feat_store.edge_feat_pb = edge_pb
    feat_store.meta = meta

    return (feat_store, graph_store), graph_store_other


def dist_link_neighbor_sampler_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    disjoint: bool = False,
):
    dist_data, graph_store_other = create_hetero_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend='gloo',
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method=f'tcp://localhost:{master_port}',
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)
    torch.distributed.barrier()

    # Create input rows and cols such that each pair belongs to a different
    # partition
    input_type_edge_index = dist_data[1]._edge_index[(input_type, 'coo')]
    # Edge from the current partition:
    row_0 = input_type_edge_index[0][0]
    col_0 = input_type_edge_index[1][0]
    # Edge from the other partition:
    input_type_edge_index_other = graph_store_other._edge_index[(input_type,
                                                                 'coo')]
    row_1 = input_type_edge_index_other[0][0]
    col_1 = input_type_edge_index_other[1][0]

    # Seed nodes:
    input_row = torch.tensor([row_0, row_1], dtype=torch.int64)
    input_col = torch.tensor([col_0, col_1], dtype=torch.int64)

    inputs = EdgeSamplerInput(
        input_id=None,
        row=input_row,
        col=input_col,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(coro=dist_sampler.edge_sample(
        inputs, dist_sampler.node_sample, data.num_nodes, disjoint))

    torch.distributed.barrier()

    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=disjoint,
    )

    # Evaluate edge sample function:
    out = edge_sample(
        inputs,
        sampler._sample,
        data.num_nodes,
        disjoint,
        node_time=None,
        neg_sampling=None,
    )

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        if disjoint:
            assert torch.equal(out_dist.batch[k].sort()[0],
                               out.batch[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]

    torch.distributed.barrier()


def dist_link_neighbor_sampler_temporal_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    disjoint: bool = False,
):
    dist_data, graph_store_other = create_hetero_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend='gloo',
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method=f'tcp://localhost:{master_port}',
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)
    torch.distributed.barrier()

    # Create input rows and cols such that each pair belongs to a different
    # partition
    input_type_edge_index = dist_data[1]._edge_index[(input_type, 'coo')]
    # Edge from the current partition:
    row_0 = input_type_edge_index[0][0]
    col_0 = input_type_edge_index[1][0]
    # Edge from the other partition:
    input_type_edge_index_other = graph_store_other._edge_index[(input_type,
                                                                 'coo')]
    row_1 = input_type_edge_index_other[0][0]
    col_1 = input_type_edge_index_other[1][0]

    # Seed nodes:
    input_row = torch.tensor([row_0, row_1], dtype=torch.int64)
    input_col = torch.tensor([col_0, col_1], dtype=torch.int64)

    inputs = EdgeSamplerInput(
        input_id=None,
        row=input_row,
        col=input_col,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(coro=dist_sampler.edge_sample(
        inputs, dist_sampler.node_sample, data.num_nodes, disjoint))

    torch.distributed.barrier()

    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=disjoint,
    )

    # Evaluate edge sample function:
    out = edge_sample(
        inputs,
        sampler._sample,
        data.num_nodes,
        disjoint,
        node_time=None,
        neg_sampling=None,
    )

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        if disjoint:
            assert torch.equal(out_dist.batch[k].sort()[0],
                               out.batch[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]

    torch.distributed.barrier()


@withPackage('pyg_lib')
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_link_neighbor_sampler_hetero(tmp_path, disjoint):
    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    world_size = 2
    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 0, port, ('v0', 'e0', 'v0'),
              disjoint),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 1, port, ('v1', 'e0', 'v0'),
              disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
