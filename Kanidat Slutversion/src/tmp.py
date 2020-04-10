states = all_states[~self.is_terminal & ~self.is_leaf]
        
        
        #...........................Check if terminal state.............................

        #Check which of the states is terminal:
        #print("checking if terminal b4 and aftr: ")
        self.is_terminal[~self.is_leaf] = np.array([np.all(state==0) for state in states)
        #place the current terminal and non terminal states in seprate arrays
        non_terminal_states = states[~self.is_terminal[~self.is_leaf]]

        terminal_indecies = np.where(self.is_terminal & ~self.is_leaf)[0]
        self.v_arr[self.is_terminal & ~self.is_leaf] = self.eval_ground_states(actions_taken, terminal_indecies)

        #Terminal states --> ej aktiva längre (används inte längre ned i trädet)
        

        # np.arrays unhashable, needs string

        '''Create n strings from n states'''
        state_strings = np.array([str(state) for state in non_terminal_states])

        # ............................If leaf node......................................
        '''av de i continuing_view --> kolla om dess stater har Q-värden här'''
        '''sedan generera alla perspektiv här...'''
        #ta bort alla de nya terminal states från leafs (om den är terminal är den ej en leafstat)
        self.is_leaf[~self.is_leaf] = self.is_terminal[~self.is_leaf]
        
        new_is_leaf = np.array([s not in self.Ns for s in state_strings], dtype=bool)
        #detta fungerar inte!!
        self.is_leaf[~self.is_leaf & ~self.is_terminal] = new_is_leaf

        leaf_states = non_terminal_states[new_is_leaf]
        non_leaf_states = non_terminal_states[~new_is_leaf]


        new_is_active = ~self.is_terminal & ~self.is_leaf

        leaf_state_strings = [str(state) for state in leaf_states]
        non_leaf_state_strings = [str(state) for state in non_leaf_states]