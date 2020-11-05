import gym
from agents.ddpg import *
from envs.env_wrapper import *
from mems.replay import *
from nets.networks import *
from matplotlib import pyplot as plt
#from BicycleRender import BicycleRender
from gym.envs.registration import register 


register(
    id='BicycleBalance-v0',
    entry_point='envs:BicycleBalanceEnv')

# Define environment name
ENV_NAME = "BicycleBalance-v0"

# Initialize the environment
#env = gym.make(ENV_NAME)
#env= ContinuousWrapper(gym.make(ENV_NAME))
env = gym.make(ENV_NAME)

# Get the action dimension and state dimension
action_dim = env.action_space.shape[0] # action_dim = 1
state_dim = env.observation_space.shape # state_dim = 5

# Initialize the network of DDPG algorithm
# online critic and target critic
critic = CriticNetwork(action_dim=action_dim, state_dim= state_dim)


# online actor and target actor
actor = ActorNetwork(action_dim=action_dim, state_dim= state_dim)
memory = Memory(1000000, state_dim, 1, 64)

def first():
    agent.bike_learning()

def second():
    agent.evaluate()

def end():
    agent.restore()
    app = BicycleRender(agent, env)
    app.run()

def only_hardware():
    agent.hardware_()

with tf.Session() as sess:
    # create the agent 
    agent = Agent_DDPG(sess, actor, critic, memory, env=env,
                 max_test_epoch=2000, b4_train=5000,
                 max_step_per_game=100, plot=True,
                 render=False, max_episode=10000, env_name=ENV_NAME,
                 OUnoise_theta=0.15, OUnoise_sigma=0.1)

    only_hardware()






