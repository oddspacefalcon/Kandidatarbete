import Edge

class Node():
    """
    Noden representerar ett tillstånd.
    Varje nod i trädet innehåller referenser till tre edges som representerar våra drag.
    Noden innehåller även referenser till child nodes.
    """

    def __init__(self, state):
        self.state = state # tillståndet som noden representerar
        self.nodes = [] # innehåller referenser till höger, mitten och vänster barnnod
        self.edges = [] # motsvarande edges

    def add_child(self, new_state):
        self.nodes.append(Node(new_state))
        self.edges.append(Edge(new_state))