import graphviz
from typing import Union
from juligrad.tensor import Tensor
from juligrad.ops import Op

def generateGraph(sink: Union[Op, Tensor]):
    def _draw_node(node):
        '''Draws / adds a single node to the graph.'''
        # Don't add duplicate nodes to the graph.
        # e.g. if we reach a node twice from its two downstream nodes, only add it once
        if f'\t{id(node)}' in dot.body: return
        
        # Add the node with the appropriate text
        if isinstance(node, Tensor):
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>shape = {node.shape}</TD></TR>
                <TR><TD>requiresGrad = {node.requiresGrad}</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">input</FONT></TD></TR>
            </TABLE>>'''
        elif isinstance(node, Op):
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>name = {type(node).__name__}</TD></TR>
                <TR><TD BGCOLOR="#c2ebff"><FONT COLOR="#004261" FACE="Courier" POINT-SIZE="12"></FONT></TD></TR>
            </TABLE>>'''
        else:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>Unknown node</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">input</FONT></TD></TR>
            </TABLE>>'''
        dot.node(str(id(node)), node_text)
            
    def _draw_edge(parent, node):
        '''Draws / adds a single directed edge to the graph (parent -> node).'''
        # Don't add duplicate edges to the graph.
        # e.g. if we reach a node twice from its two downstream nodes, only add edges to its parents once
        if f'\t{id(parent)} -> {id(node)}' in dot.body: return
        
        # Add the edge
        dot.edge(str(id(parent)), str(id(node)))
    
    def _draw_parents(node):
        '''Traverses recursively, drawing the parent at the child's step (in order to draw the edge).'''
        if isinstance(node, Tensor):
            _draw_node(node.sourceOp)
            _draw_edge(node.sourceOp, node)
            _draw_parents(node.sourceOp)

        if isinstance(node, Op):
            parents = [v for k,v in node.__dict__.items() if isinstance(v,Tensor)]
            for parent in parents:
                _draw_node(parent)
                _draw_edge(parent, node)
                _draw_parents(parent)

    dot = graphviz.Digraph(graph_attr={'rankdir': 'BT'}, node_attr={'shape': 'plaintext'})
    _draw_node(sink)     # Draw the root / output      
    _draw_parents(sink)  # Draw the rest of the graph
    return dot