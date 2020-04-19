
import numpy as np
import matplotlib.pyplot as plt

def get_data(PATH):
    with open(PATH + '\data_result.txt')as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    P_success = []
    P_error= []
    II =  []
    i = 0
    for row in  content:    
        P_e, P_s = row.split(',')
        P_e = float(P_e)
        P_s = float(P_s)
        if P_e != 0.0 and P_s != 0.0:
            P_error.append(P_e)
            P_success.append(P_s)
            II.append(i)
            i = i+1
    
    return P_success, P_error



def plot(system_size5,system_size11, system_size13, PATH5, PATH11, PATH13, plot_range):
    P_success5, P_error5 = get_data(PATH5)
    P_success11, P_error11 = get_data(PATH11)
    P_success13, P_error13 = get_data(PATH13)
     
    
    fig, ax = plt.subplots()

    ax.scatter(P_error5, P_success5,label='d = '+str(system_size5), color='steelblue', marker='o')
    ax.scatter(P_error11, P_success11,label='d = '+str(system_size11), color='green', marker='^')
    ax.scatter(P_error13, P_success13,label='d = '+str(system_size13), color='orange', marker='s')
    ax.legend(fontsize = 14)
    ax.plot(P_error5,P_success5, color='steelblue')
    ax.plot(P_error11,P_success11, color='green')
    ax.plot(P_error13,P_success13, color='orange')
    ax.set_xlim(0.005,plot_range*0.01+0.005)
    plt.xlabel('$P_e$', fontsize=14)
    plt.ylabel('$P_s$', fontsize=14)
    plt.title('Prestanda för trändade agenter')
    plt.tick_params(axis='both', labelsize=14)
    #fig.set_figwidth(10)
    plt.savefig('Results'+'/Agent_Total_Result_Plot'+'.png')
    plt.show()


#######################################
system_size5 = 5
system_size11 = 11
system_size13 = 13

PATH5  = 'Results/Main_Size_5_NN_11_steps_epoch_7_steps_7000__20_04_15__11__54__14__'
PATH11  = 'Results/Main_size_11_Size_11_ResNet18_memory_uniform__steps_7800_learning_rate_0.00025'
PATH13  = 'Results/Main_Size_13_ResNet18_memory_uniform__steps_2300_learning_rate_0.00025'

plot_range = 10 # plot from P_error = 0.01 to plot_range*0.01


plot(system_size5,system_size11, system_size13,PATH5, PATH11, PATH13, plot_range)
