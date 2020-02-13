class Edge():
    """
    Varje edgeobjekt innehåller de värden som propageras tillbaka i trädet.
    Nedanstående är de från alphagozero-artiklen.
    """
    def __init__(self):
        self.N = 0 # visit count
        self.W = 0 # total action value
        self.Q = 0 # mean action value
        self.P = 0 # prior probability of chosing this edge

    def update_values(self):
        """"
        Uppdaterar ovanstående värden
        """
        pass