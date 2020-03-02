import numpy as np

def get_syndrompos(syndrom_matrix, system_size):
    x_syndroms = np.copy(syndrom_matrix[0])
    z_syndroms = np.copy(syndrom_matrix[1])

    x_syndromPos = []
    z_syndromPos = []
    
    for i in range(system_size):
        for j in range(system_size):
            if x_syndroms[i][j] != 0:
                x_syndromPos.append((i,j))
            if z_syndrom[i][j] != 0:
                z_syndromPos.append((i,j))
    return z_syndromPos, x_syndromPos

def generate_con_matrix(x_errors, z_errors, system_size):
    nr_xerrors = x_errors.size()[0]
    nr_zerrors = z_errors.size()[0]
    conX_matrix = []
    conZ_matrix = []
    for i in range(max(nr_xerrors, nr_zerrors)):
        if(i  < nr_xerrors and j < nr_xerrors):
            conX_matrix[i] = []
        if(i  < nr_zerrors and j < nr_zerrors):
            conZ_matrix[i] = []
        for j in range(max(nr_zerrors, nr_xerrors)):
            if(i  < nr_xerrors and j < nr_xerrors):
                conX_matrix[i][j] = calc_path(x_errors[i], x_errors[j], system_size)
            if(i < nr_zerrors and j < nr_zerrors):
                conZ_matrix[i][j] = calc_path(z_errors[i], z_errors[j], system_size)
    return (conX_matrix, conZ_matrix)
    


def calc_path(pos1, pos2, size, opperation):
    x1, y1 = pos1
    x2, y2 = pos2
    x_length = abs(x2-x1) if (abs(x2-x1) < size-abs(x2-x1)) else size-abs(x2-x1)
    y_length = abs(y2-y1) if (abs(y2-y1) < size-abs(y2-y1)) else size-abs(y2-y1)
    path = np.zeros((size,size), dtype=int)
    for i in range(x_length+y_length):
        #ej korrekt nu...
        if(i < x_lentgth):
            path[x1+i][y1] = opperation+1
        else:
            path[x1+x_length-1][y1+i-x_length]
    return x_length+y_length