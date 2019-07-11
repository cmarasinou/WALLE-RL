import numpy as np

class Environment():
    ''' Generates a 3x4 world
    '''
    def __init__(self):

        self.n_states = 12
        self.h = 3 # world height
        self.w = 4 #world width
        
        # define states
        self.states_locations = [(0,0), (0,1), (0,2), (0,3),
                    (1,0), (1,1), (1,2),(1,3),
                    (2,0), (2,1), (2,2),(2,3)]
        self.states_index = np.array([i for i in range(0,self.n_states)]).reshape(self.h,self.w)
        self.current_state= 2 
        self.terminal_states = [7,11]
        self.impossible_states = [5]
        
        # define actions
        self.n_actions = 4
        # in order Left, Right, Up, Down
        self.actions = [(0,-1),(0,+1),(-1,0),(+1,0)]
        self.action_symbols = np.array(['<','>','^','v'])
        
        # define rewards
        self.rewards = np.array([-0.04, -0.04, -0.04, -0.04,
               -0.04, 0.0, -0.04,  -1.0,
               -0.04, -0.04, -0.04,   1.0])
        self.initial_reward = self.rewards[self.current_state]
        
        # define transition model
        self.transition_model = self.generate_transition_model()

        
    def step(self, action):
        '''Performs a transition given an action
        Returns:
            current_state (int)
            reward (float)
            Bool: If True episode finished and environment re-initialized
        '''
        # transition to new state
        self.current_state = np.random.choice(self.n_states, 
                     p=self.transition_model[:,action,self.current_state])
        if self.current_state in self.terminal_states:
            end_state = self.current_state
            reward = self.rewards[end_state]
            #Reinitialize
            self.__init__()
            return end_state, reward, True # Last bool indicates that episode finished
                
        return self.current_state, self.rewards[self.current_state], False
        
    def out_of_bounds(self,state_location):
        '''Determines whether given state out of bounds'''
        if state_location[0] in range(0,self.h) and state_location[1] in range(0,self.w):
            if self.states_index[state_location] in self.impossible_states:
                return True
            return False
        else:
            return True
        
    def generate_transition_model(self):
        '''Generates the transition model
        Returns:
            P (3d array, (n_states,n_actions,n_states)): Transtion model probabilites
        '''
        P = np.zeros((self.n_states,self.n_actions,self.n_states))

        for s in range(0,self.n_states):
            for a in range(0,self.n_actions):
                if s in self.terminal_states or s in self.impossible_states:
                    continue


                s_location = self.states_locations[s]

                sp_location =  (s_location[0] + self.actions[a][0],s_location[1] + self.actions[a][1])
                if self.out_of_bounds(sp_location):
                    sp_location = s_location
                sp = self.states_index[sp_location]
                prob = 0.8
                P[sp,a,s]+=prob

                opposite_actions = 1-np.abs(self.actions[a])

                sp_location =  (s_location[0] + opposite_actions[0], s_location[1] +opposite_actions[1])
                if self.out_of_bounds(sp_location):
                    sp_location = s_location
                sp = self.states_index[sp_location]
                prob = 0.1
                P[sp,a,s]+=prob


                sp_location =  (s_location[0] - opposite_actions[0], s_location[1] -opposite_actions[1])
                if self.out_of_bounds(sp_location):
                    sp_location = s_location
                sp = self.states_index[sp_location]
                prob = 0.1
                P[sp,a,s]+=prob
        return P


class Agent():
    '''Creates an agent
    '''
    def __init__(self):
        self.n_states = 12
        self.h = 3
        self.w = 4
        #define states
        self.states_locations = [(0,0), (0,1), (0,2), (0,3),
                    (1,0), (1,1), (1,2),(1,3),
                    (2,0), (2,1), (2,2),(2,3)]
        self.states_index = np.array([i for i in range(0,self.n_states)]).reshape(self.h,self.w)
        #define actions
        self.n_actions = 4
        # in order Left, Right, Up, Down
        self.actions = [(0,-1),(0,+1),(-1,0),(+1,0)]
        self.action_symbols = np.array(['<','>','^','v'])
        
        # Defining policy
        policy_symbols = np.array([['v' ,'<', '<' ,'<'],
                                   ['v', None ,'v', None],
                                   ['>' ,'>', '>', None]])
        self.policy = self.translate_policy(policy_symbols)
        
    def translate_policy(self,policy_symbols):
        '''Gets policy in symbols. Outputs policy as array of integers'''
        policy_symbols = policy_symbols.ravel()
        policy = np.zeros_like(policy_symbols, dtype=int)
        for i, symbol in enumerate(policy_symbols):
            if symbol is None:
                policy[i] = -1
            else:
                policy[i] = np.argmax(self.action_symbols==symbol)

        return policy

    def render_policy(self):
        '''Prints symbolic version of current policy'''
        policy_render = [self.action_symbols[idx] if idx in range(0,len(self.actions)) else '#'\
                    for idx in self.policy]
        policy_render = np.array(policy_render).reshape(self.h,self.w)
        print(policy_render)
        print('\n')
    
    def step(self, state):
        '''Agent decides an action given a policy
        Args:
            state (int)
        Retunrs
            action (int)
        '''
        action = self.policy[state]
        #print("Action: ",self.action_symbols[action])
        return action

