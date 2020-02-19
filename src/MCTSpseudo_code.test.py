#Mikkels kod


'''>
Funktioner: 
experience_replay(criterion, optimizer, batch_size)
get_reward()
select_action() --> förändrar toric.ne
toric.qubit_matrix --> vilken stat man är på()
toric.generatePerspective --> ger alla perspektiv som man kan ha
memory osv


Vad ska vara med här.
Borde man spara det som ett träd där man först ändast har en root node, eller ska man spara det som man gjorde i exemplet
Varför behöver man det som ett sökträd? Har man inte de olika värdena associerade med detta


'''

from .util import Transition

# main function for the Monte Carlo Tree Search

#Får använda en annan datastruktur för implementeringen av detta
class node():

    def __init__(self, data, parent):
        self.data = data
        self.parent = 
        self.children = []
        self.visits = 0
        self.

    
    def addChild(self, child):
        self.children.append(child)

    def deleteChild(self, child):
        self.children.remove(child)
    
    def addVisits():
        self.visits+=1
    
    def addTotVisits():
        self.totVisits+=1
        if(self.parent!= None):
            self.parent.addTotVisits()

class tree():


    def __init__(self, root):
        self.root = root

class MCTS():

    def __init__(self, nnet, toric, args):
        self.toric = toric
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def monte_carlo_tree_search(root):#main(mainfunktionen som går igenom det hela) 
        
        
        while resources_left(time, computational power):
            leaf = traverse(root)
            simulation_result = rollout(leaf) 
            backpropagate(leaf, simulation_result) 
            
        return best_child(root) 
    
    # function for node traversal 
    def traverse(node): 
        while fully_expanded(node): 
            #Get best actions instead by calculating the 
            node = best_uct(node) 
            
        # in case no children are present / node is terminal  
        return pick_univisted(node.children) or node  
    
    # function for the result of the simulation 
    def rollout(node): 
        while non_terminal(node): 
            node = rollout_policy(node) 
        return result(node)  
    
    # function for randomly selecting a child node 
    '''Ska vi ändast använda random policy eller ska nätvärket bestämma/nogot imellan?'''
    def rollout_policy(node): 
        return pick_random(node.children) 
    
    # function for backpropagation 
    def backpropagate(node, result): 
        if is_root(node) return
        node.stats = update_stats(node, result)  
        backpropagate(node.parent) 
    
    # function for selecting the best child 
    # node with highest number of visits 
    def best_child(node): 
        pick child with highest number of visits 