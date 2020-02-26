import numpy as np
from util import Action, Perspective
from toric_model import Toric_code
import math
import copy

class MCTS():
    def __init__(self, args, nnet, toric_code):

        '''
        Problem med denna kod:
        1. Behöver ha tillgång till 'device' för att kunna köra perspectives igenom nätvärket se #Problem 1 


        '''
        self.toric = toric_code
        self.nnet = nnet
        #memory buffer används ej här
        #self.memory_buffer = memeory_buffer
        '''
        args ska innehålla gridshift variablen från RL
        '''
        self.args = args
        self.root_state = np.copy(self.toric.qubit_matrix)
        self.actions_taken = []
        
        self.index = 0
        self.branches = {} #inte säker på om denna behövs --> använder denna iallafall inte

        #Vet ej om vi ska byta ut dessa?
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s

        #Denna används inte i våran algoritm ()
        self.Vs = {}        # stores game.getValidMoves for board s 

    def get_input_perspectives(self):
        self.toric.syndrom('next_state')
        return self.toric.generate_perspective(self.args.gridshift, self.toric.next_state)
    
    def get_actionprob(self, temp=1):
        for i in range(self.args.nr_simulations):
            print("nr simulations " + str(i))
            self.search(copy.deepcopy(self.root_state))
        
        self.toric.qubit_matrix = self.root_state
        s = str(self.root_state)

        #Ger alla positioner: [(Q, i, j), (Q, i, j), ... ] för alla perspektiv (i korrekt ordning)
        perspective_pos = Perspective(*zip(*self.get_input_perspectives())).position

        #actions = [[X, Y, Z], [X, Y, Z], ... ] --> Varje [X, Y, Z] är dessa opperatorer på ett visst perspektiv
        actions = [[str(Action(np.array(p_pos), x+1)) for x in range(3)] for p_pos in perspective_pos]

        
        pi = [[self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in perspective] for perspective in actions]
        
        if temp==0:
            bestA = np.argmax(pi)
            pi = [0]*len(pi)
            pi[bestA]=1
            return pi
        #pi = [[prob**(1./temp) for prob in perspective] for perspective in pi]
        print(np.sum(pi))
        pi = np.array(pi)

        return pi

    def UCBpuct(self, probability_matrix, actions, s):
        #s = qubit-matrisen
        #en annan fråga är ochså hur jag representerar actions här. Vanligtvis brukar detta göras med att man räknar ut alla
        #Borde hitta nogont set datatyp som låter en göra parallela acions på flera state action pairs samtidigt
        #Borde också eventuellt hitta ett bättre sätt att representera actions på ett bättre sätt
        current_Qsa = np.array([[self.Qsa[(s,str(a))] if (s, str(a)) in self.Qsa else 0 for a in opperator_actions] for opperator_actions in actions])
        current_Nsa = np.array([[self.Nsa[(s,str(a))] if (s, str(a)) in self.Nsa else 0 for a in opperator_actions] for opperator_actions in actions])
        if s not in self.Ns:
            current_Ns = 0.001
        else:
            if self.Ns[s] == 0:
                current_Ns = 0.001
            else:
                current_Ns = self.Ns[s]
        return current_Qsa + self.args.cpuct*probability_matrix*np.sqrt(current_Ns/(1+current_Nsa))


    def search(self, current_qubit_state):
        
        #vet ej om denna behövs:
        self.toric.qubit_matrix = current_qubit_state
        self.toric.syndrom('next_state')

        s = str(current_qubit_state)
        if s not in self.Es:
            if self.toric.terminal_state('next_state') == 1:#then not terminal state
                self.Es[s] = 0
            else:
                if self.toric.eval_ground_sate():
                    #Trivial loop --> gamestate won!
                    self.Es[s] = 1 #or the reward 100 i dunno
                else:
                    #non trivial loop --> game lost!
                    self.Es[s] = -1
        
        current_Es = self.Es[s]
        if current_Es != 0:
            #Backprop?
            return self.Es[s]
        array_of_perspectives = self.get_input_perspectives()
        #gör om från np[Perspective(perspective, position)] --> Perspective([perspective], [position])*
        perspectives = Perspective(*zip(*array_of_perspectives))
        input_perspectives = np.array(perspectives.perspective)
        perspective_pos = np.array(perspectives.position)
        if s not in self.Ps:
            #Problem 1
            #innan denna matas inn i nätvärket behöver denna först konverteras till en torch tensor och stoppas in i decve
            #sedan autputar nätvärket ochså policyn för detta
            #Ps ska inte innehålla v!
            #detta måste vara fel format!
            '''
            om output på nnet är [X, Y, Z, v]:
            out = nnet.feedforward(perspective_input)
            Ps[s] = [[out[i][j] for j in range(3)] for i in range(len(out))]
            #genomsnittet av alla v värden för alla perspektiv
            v = np.sum([out[i][3] for i in range(len(out))])/len(out)
            '''
            v, self.Ps[s] = self.nnet.feedforward(input_perspectives)
            
            #normalisera
            self.Ps[s]  = self.Ps[s]/np.sum(self.Ps[s])

            #är nu färdig med episoden
            self.Ns[s] = 0
            return v

        #positionen blir här en tuple: type(p_pos) = tuple, INTE array!
        #inneffektivt sätt att göra det på som jag har förstått
        actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspective_pos]
        UpperConfidence = self.UCBpuct(self.Ps[s], actions, s)

        
        #indecies_of_action = delat upp i perspektiv behövs inte
        perspective_index, action_index = np.unravel_index(np.argmax(UpperConfidence), UpperConfidence.shape)
        best_perspective = array_of_perspectives[perspective_index]

        best_action = Action(np.array(best_perspective.position), action_index+1)

        self.toric.current_state = self.toric.next_state

        #Göra om så att vi ändast behöver ha perspektiv här...
        a = str(best_action)
        
        
        if (s,a) in self.Nsa:
            self.Nsa[(s,a)] +=1
        else:
            self.Nsa[(s,a)] = 1
        
        if s in self.Ns:
            self.Ns[s]+=1
        else:
            self.Ns[s]=1

        self.toric.step(best_action)
        v = self.search(np.copy(self.toric.qubit_matrix))

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = ((self.Nsa[(s,a)]-1)*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)])

        else:
            self.Qsa[(s,a)] = v
        return v
    
    def expansion(self, curent_qubit_state):
        pass
    #backprop händer ej förren man har blivit klar med en episod  --> används ej i alpha go's version av algoritmen
    def backprop(self):
        pass