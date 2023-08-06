import graphviz
from typing import Union
from juligrad.tensor import Tensor
from juligrad.ops import Op

def generateGraph(sink: Union[Op, Tensor]):
    def _draw_node(node):
        if f'\t{id(node)}' in dot.body: return
        if isinstance(node, Tensor):
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>shape = {node.shape}</TD></TR>
                <TR><TD>requiresGrad = {node.requiresGrad}</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">Tensor</FONT></TD></TR>
            </TABLE>>'''
        elif isinstance(node, Op):
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>name = {type(node).__name__}</TD></TR>
                <TR><TD BGCOLOR="#c2ebff"><FONT COLOR="#004261" FACE="Courier" POINT-SIZE="12">Op</FONT></TD></TR>
            </TABLE>>'''
        else:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>Unknown node</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">Unknown</FONT></TD></TR>
            </TABLE>>'''
        dot.node(str(id(node)), node_text)
            
    def _draw_edge(parent, node):
        if f'\t{id(parent)} -> {id(node)}\n' in dot.body: return
        dot.edge(str(id(parent)), str(id(node)))
    
    def _draw_parents(node):
        if isinstance(node, Tensor):
            if node.sourceOp is not None:
                _draw_node(node.sourceOp)
                _draw_edge(node.sourceOp, node)
                _draw_parents(node.sourceOp)

        if isinstance(node, Op):
            parents = [v for k,v in node.__dict__.items() if isinstance(v,Tensor)]
            for parent in parents:
                _draw_node(parent)
                _draw_edge(parent, node)
                _draw_parents(parent)

    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'}, node_attr={'shape': 'plaintext'})
    _draw_node(sink)       
    _draw_parents(sink)  
    return dot