# Mikkels kod

import numpy as np

from toric_model import Toric_code
from toric_model import Action
from toric_model import Perspective

import os

# vad gör denna kod mer excakt
system_size = 3
nr_errors = 2
toric = Toric_code(system_size)
toric.generate_n_random_errors(nr_errors)


def generate_plaquette_string():
    string = ""
    top = "v---Q1--"
    middle = "|       "
    bottom = "Q0  p   "
    for i in range(system_size):
        for j in range(system_size):
            string += top
        string += "+\n"
        for j in range(system_size):
            string += middle
        string += "|\n"
        for j in range(system_size):
            string += bottom
        string += "|\n"
        for j in range(system_size):
            string += middle
        string += "|\n"
        if(i == system_size-1):
            for j in range(system_size):
                string += "+---+---"
            string += "+"

    return string



def syndrom_tostring():
    os.system('cls')
    myString = generate_plaquette_string()
    syndrom = toric.current_state
    vertex_syndrom = syndrom[0][:][:]
    plaquette_syndrom = syndrom[1][:][:]
    strList = list(myString)
    for i in range(system_size):
        for j in range(system_size):
            strList[myString.find('v')] = str(vertex_syndrom[i][j])
            strList[myString.find('p')] = str(plaquette_syndrom[i][j])
            myString = ''.join(strList)
    

    print(myString)


state = toric.current_state
toric.plot_toric_code('hej')


# os.system('cls')
while(toric.terminal_state(toric.current_state == 1)):
    os.system('cls')
    print('\n\n')
    toric.syndrom('state')
    syndrom_tostring()
    #print("q-bit matrix:\n" + str(toric.get_qubit_matrix()))
    #print("Syndrome (X-plaquette followed by Z-vertex errors): \n" + str(toric.get_syndrom()))

    txtinput = input(
        "\nGive action: q_matrix,ypos,xpos,action(1=x,2=y,3=z):\t")
    input_sp = txtinput.split(',')

    pos = [int(input_sp[0]), int(input_sp[1]), int(input_sp[2])]
    action = int(input_sp[3])

    myAction = Action(pos, action)
    toric.step(myAction)

toric.eval_ground_state()
if(toric.ground_state == True):
    print('You win!')
else:
    print("You Lose!")
