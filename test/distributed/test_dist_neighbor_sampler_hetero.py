import atexit
import socket
from typing import Optional

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
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.sampler.neighbor_sampler import node_sample
from torch_geometric.testing import withPackage


def create_hetero_data(tmp_path: str, rank: int,
                       time_attr: Optional[str] = None):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feature_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)
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

    feature_store.partition_idx = partition_idx
    feature_store.num_partitions = num_partitions
    feature_store.node_feat_pb = node_pb
    feature_store.edge_feat_pb = edge_pb
    feature_store.meta = meta

    if time_attr == 'time':  # Create node-level time data:
        feature_store.put_tensor(
            tensor=torch.ones(len(node_pb['v0']), dtype=torch.int64),
            group_name='v0',
            attr_name=time_attr,
        )
        feature_store.put_tensor(
            tensor=torch.full((len(node_pb['v1']), ), 2, dtype=torch.int64),
            group_name='v1',
            attr_name=time_attr,
        )
    elif time_attr == 'edge_time':  # Create edge-level time data:
        for i, (attr,
                edge_index) in enumerate(graph_store._edge_index.items()):
            feature_store.put_tensor(
                tensor=torch.full((edge_index.size(1), ), i,
                                  dtype=torch.int64),
                group_name=attr[0],
                attr_name=time_attr,
            )

    return feature_store, graph_store


def dist_neighbor_sampler_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    disjoint: bool = False,
):
    dist_data = create_hetero_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=[-1, -1],
        shuffle=False,
        disjoint=disjoint,
    )
    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # Create inputs nodes such that each belongs to a different partition
    node_pb_list = dist_data[1].node_pb[input_type].tolist()
    node_0 = node_pb_list.index(0)
    node_1 = node_pb_list.index(1)

    input_node = torch.tensor([node_0, node_1], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1, -1],
        disjoint=disjoint,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        if disjoint:
            assert torch.equal(out_dist.batch[k].sort()[0],
                               out.batch[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]


def dist_neighbor_sampler_temporal_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    seed_time: torch.tensor = None,
    temporal_strategy: str = 'uniform',
    time_attr: str = 'time',
):
    dist_data = create_hetero_data(tmp_path, rank, time_attr)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=[-1, -1],
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # Create inputs nodes such that each belongs to a different partition
    node_pb_list = dist_data[1].node_pb[input_type].tolist()
    node_0 = node_pb_list.index(0)
    node_1 = node_pb_list.index(1)

    input_node = torch.tensor([node_0, node_1], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        time=seed_time,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1, -1],
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        assert torch.equal(out_dist.batch[k].sort()[0], out.batch[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]


@withPackage('pyg_lib')
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_neighbor_sampler_hetero(tmp_path, disjoint):
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
        target=dist_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@withPackage('pyg_lib')
@pytest.mark.parametrize('seed_time', [None, [0, 0], [1, 1], [3, 3]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_neighbor_sampler_temporal_hetero(tmp_path, seed_time,
                                               temporal_strategy):
    if seed_time is not None:
        seed_time = torch.tensor(seed_time)

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

    # Add time information to data after generating partitions (support TBD)
    data['v0'].time = torch.ones(data.num_nodes_dict['v0'], dtype=torch.int64)
    data['v1'].time = torch.full((data.num_nodes_dict['v1'], ), 2,
                                 dtype=torch.int64)

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', seed_time,
              temporal_strategy, 'time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', seed_time,
              temporal_strategy, 'time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@withPackage('pyg_lib')
@pytest.mark.parametrize('seed_time', [[0, 0], [1, 2], [3, 3]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_neighbor_sampler_edge_level_temporal_hetero(
        tmp_path, seed_time, temporal_strategy):
    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    world_size = 2
    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=10,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    # Add time information to data after generating partitions (support TBD)
    for i, edge_type in enumerate(data.edge_types):
        data[edge_type].edge_time = torch.full(
            (data[edge_type].edge_index.size(1), ), i, dtype=torch.int64)

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', seed_time,
              temporal_strategy, 'edge_time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', seed_time,
              temporal_strategy, 'edge_time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
