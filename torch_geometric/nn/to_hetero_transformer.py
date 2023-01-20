import copy
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

import torch_geometric
from torch_geometric.nn.dense import HeteroLinear
from torch_geometric.nn.fx import Transformer, get_submodule
from torch_geometric.typing import EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils.hetero import (
    check_add_self_loops,
    get_unused_node_types,
)

try:
    from torch.fx import Graph, GraphModule, Node
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphModule, Graph, Node = 'GraphModule', 'Graph', 'Node'


def get_dict(mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return mapping if mapping is not None else {}


def to_hetero(module: Module, metadata: Metadata, aggr: str = "sum",
              input_map: Optional[Dict[str, str]] = None,
              debug: bool = False) -> GraphModule:
    r"""Converts a homogeneous GNN model into its heterogeneous equivalent in
    which node representations are learned for each node type in
    :obj:`metadata[0]`, and messages are exchanged between each edge type in
    :obj:`metadata[1]`, as denoted in the `"Modeling Relational Data with Graph
    Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper:

    .. code-block:: python

        import torch
        from torch_geometric.nn import SAGEConv, to_hetero

        class GNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv((-1, -1), 32)
                self.conv2 = SAGEConv((32, 32), 32)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                return x

        model = GNN()

        node_types = ['paper', 'author']
        edge_types = [
            ('paper', 'cites', 'paper'),
            ('paper', 'written_by', 'author'),
            ('author', 'writes', 'paper'),
        ]
        metadata = (node_types, edge_types)

        model = to_hetero(model, metadata)
        model(x_dict, edge_index_dict)

    where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
    hold node features and edge connectivity information for each node type and
    edge type, respectively.

    The below illustration shows the original computation graph of the
    homogeneous model on the left, and the newly obtained computation graph of
    the heterogeneous model on the right:

    .. figure:: ../_figures/to_hetero.svg
      :align: center
      :width: 90%

      Transforming a model via :func:`to_hetero`.

    Here, each :class:`~torch_geometric.nn.conv.MessagePassing` instance
    :math:`f_{\theta}^{(\ell)}` is duplicated and stored in a set
    :math:`\{ f_{\theta}^{(\ell, r)} : r \in \mathcal{R} \}` (one instance for
    each relation in :math:`\mathcal{R}`), and message passing in layer
    :math:`\ell` is performed via

    .. math::

        \mathbf{h}^{(\ell)}_v = \bigoplus_{r \in \mathcal{R}}
        f_{\theta}^{(\ell, r)} ( \mathbf{h}^{(\ell - 1)}_v, \{
        \mathbf{h}^{(\ell - 1)}_w : w \in \mathcal{N}^{(r)}(v) \}),

    where :math:`\mathcal{N}^{(r)}(v)` denotes the neighborhood of :math:`v \in
    \mathcal{V}` under relation :math:`r \in \mathcal{R}`, and
    :math:`\bigoplus` denotes the aggregation scheme :attr:`aggr` to use for
    grouping node embeddings generated by different relations
    (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`).

    Args:
        module (torch.nn.Module): The homogeneous model to transform.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        aggr (str, optional): The aggregation scheme to use for grouping node
            embeddings generated by different relations
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"sum"`)
        input_map (Dict[str, str], optional): A dictionary holding information
            about the type of input arguments of :obj:`module.forward`.
            For example, in case :obj:`arg` is a node-level argument, then
            :obj:`input_map['arg'] = 'node'`, and
            :obj:`input_map['arg'] = 'edge'` otherwise.
            In case :obj:`input_map` is not further specified, will try to
            automatically determine the correct type of input arguments.
            (default: :obj:`None`)
        debug (bool, optional): If set to :obj:`True`, will perform
            transformation in debug mode. (default: :obj:`False`)
    """
    transformer = ToHeteroTransformer(module, metadata, aggr, input_map, debug)
    return transformer.transform()


class ToHeteroModule(Module):
    aggrs = {
        'sum': torch.add,
        # For 'mean' aggregation, we first sum up all feature matrices, and
        # divide by the number of matrices in a later step.
        'mean': torch.add,
        'max': torch.max,
        'min': torch.min,
        'mul': torch.mul,
    }

    def __init__(
        self,
        module: Module,
        metadata: Metadata,
        aggr: str = 'sum',
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.aggr = aggr
        assert len(metadata) == 2
        assert aggr in self.aggrs.keys()
        # check wether module is linear
        self.is_lin = is_linear(module)
        # check metadata[0] has node types
        # check metadata[1] has edge types if module is MessagePassing
        assert len(metadata[0]) > 0 and (len(metadata[1]) > 0
                                         or not self.is_lin)
        if self.is_lin:
            # make HeteroLinear layer based on metadata
            if isinstance(module, torch.nn.Linear):
                in_ft = module.in_features
                out_ft = module.out_features
            else:
                in_ft = module.in_channels
                out_ft = module.out_channels
            heteromodule = HeteroLinear(in_ft, out_ft, len(
                self.node_types)).to(list(module.parameters())[0].device)
            heteromodule.reset_parameters()
        else:
            # copy MessagePassing module for each edge type
            unused_node_types = get_unused_node_types(*metadata)
            if len(unused_node_types) > 0:
                warnings.warn(
                    f"There exist node types ({unused_node_types}) whose "
                    f"representations do not get updated during message passing "
                    f"as they do not occur as destination type in any edge type. "
                    f"This may lead to unexpected behaviour.")
            heteromodule = {}
            for edge_type in self.edge_types:
                heteromodule[edge_type] = copy.deepcopy(module)
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                elif sum([p.numel() for p in module.parameters()]) > 0:
                    warnings.warn(
                        f"'{module}' will be duplicated, but its parameters"
                        f"cannot be reset. To suppress this warning, add a"
                        f"'reset_parameters()' method to '{module}'")

        self.heteromodule = heteromodule

    def fused_forward(self, x: Tensor, edge_index: OptTensor = None,
                      node_type: OptTensor = None,
                      edge_type: OptTensor = None) -> Tensor:
        r"""
        Args:
            x: The input node features. :obj:`[num_nodes, in_channels]`
                node feature matrix.
            edge_index (LongTensor): The edge indices.
            node_type: The one-dimensional node type/index for each node in
                :obj:`x`.
            edge_type: The one-dimensional edge type/index for each edge in
                :obj:`edge_index`.
        """
        # (TODO) Add Sparse Tensor support
        if self.is_lin:
            # call HeteroLinear layer
            out = self.heteromodule(x, node_type)
        else:
            # iterate over each edge type
            for j, module in enumerate(self.heteromodule.values()):
                e_idx_type_j = edge_index[:, edge_type == j]
                o_j = module(x, e_idx_type_j)
                if j == 0:
                    out = o_j
                else:
                    out += o_j
        return out

    def dict_forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Dict[NodeType, Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
        """
        # (TODO) Add Sparse Tensor support
        if self.is_lin:
            # fuse inputs
            x = torch.cat([x_j for x_j in x_dict.values()])
            size_list = [feat.shape[0] for feat in x_dict.values()]
            sizes = torch.tensor(size_list, dtype=torch.long, device=x.device)
            node_type = torch.arange(len(sizes), device=x.device)
            node_type = node_type.repeat_interleave(sizes)
            # HeteroLinear layer
            o = self.heteromodule(x, node_type)
            o_dict = {
                key: o_i.squeeze()
                for key, o_i in zip(x_dict.keys(), o.split(size_list))
            }
        else:
            o_dict = {}
            # iterate over each edge_type
            for j, (etype_j, module) in enumerate(self.heteromodule.items()):
                e_idx_type_j = edge_index_dict[etype_j]
                src_node_type_j = etype_j[0]
                dst_node_type_j = etype_j[-1]
                o_j = module(x_dict[src_node_type_j], e_idx_type_j)
                if dst_node_type_j not in o_dict.keys():
                    o_dict[dst_node_type_j] = o_j
                else:
                    o_dict[dst_node_type_j] += o_j
        return o_dict

    def forward(
        self,
        x: Union[Dict[NodeType, Tensor], Tensor],
        edge_index: Optional[Union[Dict[EdgeType, Tensor], Tensor]] = None,
        node_type: OptTensor = None,
        edge_type: OptTensor = None,
    ) -> Union[Dict[NodeType, Tensor], Tensor]:
        r"""
        Args:
            x (Dict[str, Tensor] or Tensor): A dictionary holding node feature
                information for each individual node type or the same
                features combined into one tensor.
            edge_index (Dict[Tuple[str, str, str], Tensor] or Tensor):
                A dictionary holding graph connectivity information for
                each individual edge type or the same values combined
                into one tensor.
            node_type: The one-dimensional relation type/index for each node in
                :obj:`x` if it is provided as a single tensor.
                Should be only :obj:`None` in case :obj:`x` is of type
                Dict[str, Tensor].
                (default: :obj:`None`)
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index` if it is provided as a single tensor.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                Dict[Tuple[str, str, str], Tensor].
                (default: :obj:`None`)
        """
        # check if x is passed as a dict or fused
        if isinstance(x, Dict):
            # check what inputs to pass
            if self.is_lin:
                return self.dict_forward(x)
            else:
                if not isinstance(edge_index, Dict):
                    raise TypeError("If x is provided as a dictionary, \
                        edge_index must be as well")
                return self.dict_forward(x, edge_index_dict=edge_index)
        else:
            if self.is_lin:
                if node_type is None:
                    raise ValueError('If x is a single tensor, \
                        node_type argument must be provided.')
                return self.fused_forward(x, node_type=node_type)
            else:
                if not isinstance(edge_index, Tensor):
                    raise TypeError("If x is provided as a Tensor, \
                        edge_index must be as well")
                if edge_type is None:
                    raise ValueError(
                        'If x and edge_indices are single tensors, \
                        node_type and edge_type arguments must be provided.')
                return self.fused_forward(x, edge_index=edge_index,
                                          edge_type=edge_type)


class ToHeteroTransformer(Transformer):

    aggrs = {
        'sum': torch.add,
        # For 'mean' aggregation, we first sum up all feature matrices, and
        # divide by the number of matrices in a later step.
        'mean': torch.add,
        'max': torch.max,
        'min': torch.min,
        'mul': torch.mul,
    }

    def __init__(
        self,
        module: Module,
        metadata: Metadata,
        aggr: str = 'sum',
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        super().__init__(module, input_map, debug)

        self.metadata = metadata
        self.aggr = aggr
        assert len(metadata) == 2
        assert len(metadata[0]) > 0 and len(metadata[1]) > 0
        assert aggr in self.aggrs.keys()

        self.validate()

    def validate(self):
        unused_node_types = get_unused_node_types(*self.metadata)
        if len(unused_node_types) > 0:
            warnings.warn(
                f"There exist node types ({unused_node_types}) whose "
                f"representations do not get updated during message passing "
                f"as they do not occur as destination type in any edge type. "
                f"This may lead to unexpected behaviour.")

        names = self.metadata[0] + [rel for _, rel, _ in self.metadata[1]]
        for name in names:
            if not name.isidentifier():
                warnings.warn(
                    f"The type '{name}' contains invalid characters which "
                    f"may lead to unexpected behaviour. To avoid any issues, "
                    f"ensure that your types only contain letters, numbers "
                    f"and underscores.")

    def placeholder(self, node: Node, target: Any, name: str):
        # Adds a `get` call to the input dictionary for every node-type or
        # edge-type.
        if node.type is not None:
            Type = EdgeType if self.is_edge_level(node) else NodeType
            node.type = Dict[Type, node.type]

        self.graph.inserting_after(node)

        dict_node = self.graph.create_node('call_function', target=get_dict,
                                           args=(node, ), name=f'{name}_dict')
        self.graph.inserting_after(dict_node)

        for key in self.metadata[int(self.is_edge_level(node))]:
            out = self.graph.create_node('call_method', target='get',
                                         args=(dict_node, key, None),
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def get_attr(self, node: Node, target: Any, name: str):
        raise NotImplementedError

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        # Add calls to edge type-wise `MessagePassing` modules and aggregate
        # the outputs to node type-wise embeddings afterwards.

        module = get_submodule(self.module, target)
        check_add_self_loops(module, self.metadata[1])

        # Group edge-wise keys per destination:
        key_name, keys_per_dst = {}, defaultdict(list)
        for key in self.metadata[1]:
            keys_per_dst[key[-1]].append(key)
            key_name[key] = f'{name}__{key[-1]}{len(keys_per_dst[key[-1]])}'

        for dst, keys in dict(keys_per_dst).items():
            # In case there is only a single edge-wise connection, there is no
            # need for any destination-wise aggregation, and we can already set
            # the intermediate variable name to the final output name.
            if len(keys) == 1:
                key_name[keys[0]] = f'{name}__{dst}'
                del keys_per_dst[dst]

        self.graph.inserting_after(node)
        for key in self.metadata[1]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=key_name[key])
            self.graph.inserting_after(out)

        # Perform destination-wise aggregation.
        # Here, we aggregate in pairs, popping the first two elements of
        # `keys_per_dst` and append the result to the list.
        for dst, keys in keys_per_dst.items():
            queue = deque([key_name[key] for key in keys])
            i = 1
            while len(queue) >= 2:
                key1, key2 = queue.popleft(), queue.popleft()
                args = (self.find_by_name(key1), self.find_by_name(key2))

                new_name = f'{name}__{dst}'
                if self.aggr == 'mean' or len(queue) > 0:
                    new_name = f'{new_name}_{i}'

                out = self.graph.create_node('call_function',
                                             target=self.aggrs[self.aggr],
                                             args=args, name=new_name)
                self.graph.inserting_after(out)
                queue.append(new_name)
                i += 1

            if self.aggr == 'mean':
                key = queue.popleft()
                out = self.graph.create_node(
                    'call_function', target=torch.div,
                    args=(self.find_by_name(key), len(keys_per_dst[dst])),
                    name=f'{name}__{dst}')
                self.graph.inserting_after(out)

    def call_global_pooling_module(self, node: Node, target: Any, name: str):
        # Add calls to node type-wise `GlobalPooling` modules and aggregate
        # the outputs to graph type-wise embeddings afterwards.
        self.graph.inserting_after(node)
        for key in self.metadata[0]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=f'{node.name}__{key2str(key)}')
            self.graph.inserting_after(out)

        # Perform node-wise aggregation.
        queue = deque(
            [f'{node.name}__{key2str(key)}' for key in self.metadata[0]])
        i = 1
        while len(queue) >= 2:
            key1, key2 = queue.popleft(), queue.popleft()
            args = (self.find_by_name(key1), self.find_by_name(key2))
            out = self.graph.create_node('call_function',
                                         target=self.aggrs[self.aggr],
                                         args=args, name=f'{name}_{i}')
            self.graph.inserting_after(out)
            queue.append(f'{name}_{i}')
            i += 1

        if self.aggr == 'mean':
            key = queue.popleft()
            out = self.graph.create_node(
                'call_function', target=torch.div,
                args=(self.find_by_name(key), len(self.metadata[0])),
                name=f'{name}_{i}')
            self.graph.inserting_after(out)
        self.replace_all_uses_with(node, out)

    def call_module(self, node: Node, target: Any, name: str):
        print("node:", node)
        print("target:", target)
        print("name:", name)
        if self.is_graph_level(node):
            return
        if hasattr(self.module, name) and is_linear(getattr(self.module,
                                                            name)):
            #insert a HeteroLinear HeteroModule instead
            self.graph.inserting_after(node)
            kwargs_dict = {}
            args_dict = {}
            for key in self.metadata[int(self.is_edge_level(node))]:
                args, kwargs = self.map_args_kwargs(node, key)
                print(type(args[0]))
                print(args[0])
                args_dict[key] = args[0]
                kwargs_dict.update(kwargs)
            out = self.graph.create_node('call_module', target=f'{target}',
                                         args=(args_dict, ),
                                         kwargs=kwargs_dict, name=f'{name}')
            self.graph.inserting_after(out)
        else:
            # Add calls to node type-wise or edge type-wise modules.
            self.graph.inserting_after(node)
            for key in self.metadata[int(self.is_edge_level(node))]:
                args, kwargs = self.map_args_kwargs(node, key)
                out = self.graph.create_node('call_module',
                                             target=f'{target}.{key2str(key)}',
                                             args=args, kwargs=kwargs,
                                             name=f'{name}__{key2str(key)}')
                self.graph.inserting_after(out)

    def call_method(self, node: Node, target: Any, name: str):
        if self.is_graph_level(node):
            return

        # Add calls to node type-wise or edge type-wise methods.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.is_edge_level(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_method', target=target,
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def call_function(self, node: Node, target: Any, name: str):
        if self.is_graph_level(node):
            return

        # Add calls to node type-wise or edge type-wise functions.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.is_edge_level(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_function', target=target,
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def output(self, node: Node, target: Any, name: str):
        # Replace the output by dictionaries, holding either node type-wise or
        # edge type-wise data.
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node):
                if self.is_graph_level(value):
                    return value
                return {
                    key: self.find_by_name(f'{value.name}__{key2str(key)}')
                    for key in self.metadata[int(self.is_edge_level(value))]
                }
            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        if node.type is not None and isinstance(node.args[0], Node):
            output = node.args[0]
            if self.is_node_level(output):
                node.type = Dict[NodeType, node.type]
            elif self.is_edge_level(output):
                node.type = Dict[EdgeType, node.type]
        else:
            node.type = None

        node.args = (_recurse(node.args[0]), )

    def init_submodule(self, module: Module, target: str) -> Module:
        # Replicate each module for each node type or edge type.
        has_node_level_target = bool(
            self.find_by_target(f'{target}.{key2str(self.metadata[0][0])}'))
        has_edge_level_target = bool(
            self.find_by_target(f'{target}.{key2str(self.metadata[1][0])}'))

        if not has_node_level_target and not has_edge_level_target:
            return module
        module_is_lin = is_linear(module)
        if module_is_lin:
            return ToHeteroModule(module, self.metadata, self.aggr)
        else:
            module_dict = torch.nn.ModuleDict()
            for key in self.metadata[int(has_edge_level_target)]:

                module_dict[key2str(key)] = copy.deepcopy(module)
                if len(self.metadata[int(has_edge_level_target)]) <= 1:
                    continue
                if hasattr(module, 'reset_parameters'):
                    module_dict[key2str(key)].reset_parameters()
                elif sum([p.numel() for p in module.parameters()]) > 0:
                    warnings.warn(
                        f"'{target}' will be duplicated, but its parameters "
                        f"cannot be reset. To suppress this warning, add a "
                        f"'reset_parameters()' method to '{target}'")

            return module_dict

    # Helper methods ##########################################################

    def map_args_kwargs(self, node: Node,
                        key: Union[NodeType, EdgeType]) -> Tuple[Tuple, Dict]:
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node):
                out = self.find_by_name(f'{value.name}__{key2str(key)}')
                if out is not None:
                    return out
                elif isinstance(key, tuple) and key[0] == key[-1]:
                    name = f'{value.name}__{key2str(key[0])}'
                    return self.find_by_name(name)
                elif isinstance(key, tuple) and key[0] != key[-1]:
                    return (
                        self.find_by_name(f'{value.name}__{key2str(key[0])}'),
                        self.find_by_name(f'{value.name}__{key2str(key[-1])}'),
                    )
                else:
                    raise NotImplementedError
            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        args = tuple(_recurse(v) for v in node.args)
        kwargs = {k: _recurse(v) for k, v in node.kwargs.items()}
        return args, kwargs


def key2str(key: Union[NodeType, EdgeType]) -> str:
    key = '__'.join(key) if isinstance(key, tuple) else key
    return key.replace(' ', '_').replace('-', '_').replace(':', '_')


def is_linear(module):
    return isinstance(module, torch.nn.Linear) or isinstance(
        module, torch_geometric.nn.dense.Linear)
