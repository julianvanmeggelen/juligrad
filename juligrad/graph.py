import networkx as nx
from juligrad.ops import LazyOp
from juligrad.tensor import Tensor, LazyTensor, TensorLike

def constructGraph(sink: LazyTensor):
    if not sink.realized:
        sink.realize()
    G = nx.DiGraph()
    def label(sink):
        return f"{type(sink).__name__}_{id(sink)}"
    def traverse(sink: TensorLike):
        #G.add_node(id(sink))
        if type(sink) is LazyTensor:
            G.add_edge(label(sink.sourceOp),label(sink))
            traverse(sink.sourceOp)
        elif isinstance(sink, LazyOp):
            for parent in sink.realizedDependencies:
                G.add_edge(label(parent),label(sink))
                traverse(parent)
    traverse(sink)
    return G
