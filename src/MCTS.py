import Node
import Edge

class MCTS():
    """
    Innehåller mcts-metoderna och huvudmetoden som skapar trädet mha Node- och Edge-klasserna
    """

    def __init__(self, number_of_levels, root_state):
        self.number_of_levels = number_of_levels
        self.root_node = Node(root_state)
        self.current_node = root_node
        self.current_level = 1

    def UCB(self, root_node_state, action):
        pass

    # De fyra faserna:

    def selection(self, current_node):
        """
        Börja med rotnoden och gå ner i trädet med hjälp av UCB-funktionen
        tills det att en nod som inte har några barn nås.
        """
        pass

    def expand(self):
        """
        Om noden inte är terminal skapas tre nya tillstånd, för X, Y och Z.
        Välj sedan (slumpmässigt) den första av dessa. Öka current_level med 1
        """
        pass


    def roll_out(self):
        """
        Kör klart episoden från den nuvarande noden genom att bara välja den med maximalt
        Q värde från nätverket.
        """
        return terminal_value

    def backprop(self, terminal_value):
        """
        Uppdatera vikterna hos alla edges längs den gångna vägen.
        """
        pass


    # Skapa trädet
    def make_tree(self):

        terminal = False # 0 eller 1 som dom?
        
        while not terminal and self.current_level <= self.number_of_levels:
            if len(current_node.nodes) == 0:
                # om noden vi är på inte har några barnnoder skapar vi nya tillstånd och kör roll_out
                self.expand()
                terminal_value = self.roll_out()
                self.backprop(terminal_value)
            else:
                # annars går vi genom trädet tills nod utan barnnoder nås
                self.selection()