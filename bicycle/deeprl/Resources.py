"""
THIS IS THE PART OF CODE WHICH DEFINES NOISE AND THE REPLAY BUFFER (initializing and storing the transition values
of SARS)
"""

# Import the libraries required for processing
import numpy as np
import random


# This class is to define the noise (Ornstein-Uhlenbeck process) which is required for exploration
class OUNoise(object):

    # Defining of the variables according to the Ornstein-Unlenbeck process and required values selected are
    # accordingly defined in the paper
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=0.001, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    # Function call to return the updated x
    def __call__(self):
        x = self.prev_x + self.theta*(self.mu - self.prev_x)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.prev_x = x
        return x

    # Reset the parameter to the initial conditions or defining zeros if no value is present
    def reset(self):
        if self.x0 is not None:
            self.prev_x = self.x0
        else:
            self.prev_x = np.zeros_like(self.mu)


# This class has functions to initialize the parameters, storing the S-A-R-S'-A' for every checkpoint
# used during the process for back-propagation learning) and sample func which loads the parameters and is randomly
# selected
class ReplayBuffer(object):

    # Parameters initialization according to the Memory size
    def __init__(self, max_size, input_shape, n_act):
        self.memory_size = max_size  # memory size to be allotted
        self.memory_cntr = 0  # saving the most recent memory used
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, n_act))
        self.reward_memory = np.zeros(self.memory_size)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape))
        # self.new_action_mem = np.zeros(self.memory-size, )
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    # Storing the parameter values during transition
    def transition(self, state, action, reward, new_state, flag_complete):
        index = self.memory_cntr % self.memory_size  # if counter < memory size if returns - memory counter value and
        # if it goes more it starts again
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        #print (flag_complete)
        self.terminal_memory[index] = 1 - flag_complete  # flag used to not store the rewards after episode is done
        self.memory_cntr += 1

    # Sample buffer
    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_cntr, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        batch_states = self.state_memory[batch]
        batch_actions = self.action_memory[batch]
        batch_rewards = self.reward_memory[batch]
        batch_new_states = self.new_state_memory[batch]
        batch_terminal = self.terminal_memory[batch]

        return batch_states, batch_actions, batch_rewards, batch_new_states, batch_terminal

    def clear_or_reset(self):
        self.__init__()
