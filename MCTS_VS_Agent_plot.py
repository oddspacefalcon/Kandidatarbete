
import numpy as np
import matplotlib.pyplot as plt


def get_data(PATH):
    with open(PATH + '\data_result2.txt')as f:
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

def plot(system_size5, PATH5, PATH_MCTS, plot_range):
    P_success5, P_error5 = get_data(PATH5)
    P_success_MCTS, P_error_MCTS = get_data(PATH_MCTS)
     
    fig, ax = plt.subplots()
    #matplotlib inline
    #ax.scatter(P_error_MCTS, P_success_MCTS,s=100, label='MCTS rollout', color='green', marker='D')
    #ax.scatter(P_error5, P_success5,s=100, label='Tränat nätverk', color='steelblue', marker='o')
    ax.plot(P_error_MCTS,P_success_MCTS, '--', color='steelblue', label='MCTS rollout')
    ax.plot(P_error5,P_success5, color='steelblue', label='Tränat nätverk')
    leg = ax.legend(fontsize = 24);
    ax.set_xlim(0.005,plot_range*0.01+0.005)
    plt.xlabel('$P_e$', fontsize=24)
    plt.ylabel('$P_s$', fontsize=24)
    plt.title('Jämförelse MCTS med rollout och tränat nätverk för d = 5', fontsize=24)
    plt.tick_params(axis='both', labelsize=24)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    #plt.savefig('Results'+'/Agent_Total_Result_Plot'+'.png')
    plt.show()


#######################################
system_size5 = 5
system_size7 = 7
system_size9 = 9
system_size11 = 11
system_size13 = 13

PATH5  = 'Results/Main_Size_5_NN_11_steps_epoch_7_steps_7000__20_04_15__11__54__14__'
PATH_MCTS = 'Results/MCTS_prediction__20_04_19__22__51__26__'


plot_range = 10 # plot from P_error = 0.01 to plot_range*0.01


plot(system_size5, PATH5, PATH_MCTS, plot_range)
