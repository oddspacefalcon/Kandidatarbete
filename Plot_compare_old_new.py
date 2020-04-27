
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

def plot(system_size5, system_size7, system_size9, PATH5, PATH7, PATH9, plot_range):
    P_success5, P_error5 = get_data(PATH5)
    P_success7, P_error7 = get_data(PATH7)
    P_success9, P_error9 = get_data(PATH9)     
    
    fig, ax = plt.subplots()

    #ax.scatter(P_error5, P_success5,s=100, label='d = '+str(system_size5), color='steelblue', marker='o')
    #ax.scatter(P_error7, P_success7,s=100, label='d = '+str(system_size7), color='green', marker='D')
    #ax.scatter(P_error9, P_success9,s=100, label='d = '+str(system_size9), color='orange', marker='X')
    
    ax.plot(P_error5,P_success5, label='d = '+str(system_size5), color='steelblue')
    ax.plot(P_error7,P_success7, label='d = '+str(system_size7), color='green')
    ax.plot(P_error9,P_success9, label='d = '+str(system_size9), color='orange')
    ax.legend(fontsize = 24)
    #ax.set_xlim(0.005,plot_range*0.01+0.005)
    plt.xlabel('$P_e$', fontsize=24)
    plt.ylabel('$P_s$', fontsize=24)
    plt.title('Prestanda för trändade agenter', fontsize=24)
    plt.tick_params(axis='both', labelsize=24)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    #plt.savefig('Results'+'/Agent_Total_Result_Plot'+'.png')
    plt.show()

def plot2(system_size11, system_size13, PATH11, PATH13, plot_range):
    P_success11, P_error11 = get_data(PATH11)
    P_success13, P_error13 = get_data(PATH13)
    
    fig, ax = plt.subplots()

    #ax.scatter(P_error11, P_success11,s=100, label='d = '+str(system_size11), color='firebrick', marker='o')
    #ax.scatter(P_error13, P_success13,s=100, label='d = '+str(system_size13), color='saddlebrown', marker='D')
    
    ax.plot(P_error11,P_success11, label='d = '+str(system_size11), color='firebrick')
    ax.plot(P_error13,P_success13, label='d = '+str(system_size13), color='saddlebrown')
    ax.legend(fontsize = 24)
    ax.set_xlim(0.005,plot_range*0.01+0.005)
    plt.xlabel('$P_e$', fontsize=24)
    plt.ylabel('$P_s$', fontsize=24)
    plt.title('Prestanda för trändade agenter', fontsize=24)
    plt.tick_params(axis='both', labelsize=24)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    #plt.savefig('Results'+'/Agent_Total_Result_Plot'+'.png')
    plt.show()
#######################################
system_size5 = 5
system_size7 = 7
system_size9 = 9


PATH5  = 'Old_RL_Data/d=5'
PATH7 = 'Old_RL_Data/d=7'
PATH9 = 'Old_RL_Data/d=9'

plot_range = 20 # plot from P_error = 0.01 to plot_range*0.01

#plot(system_size5, system_size7, system_size9, PATH5, PATH7, PATH9, plot_range)

system_size11 = 11
system_size13 = 13
PATH11 = 'Old_RL_Data/New_d=11'
PATH13 = 'Old_RL_Data/New_d=13'


plot2(system_size11, system_size13, PATH11, PATH13, plot_range)