import gym
import numpy as np
from matplotlib import pyplot as plt
#from BicycleRender import BicycleRender
from gym.envs.registration import register 
import time

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
#critic = CriticNetwork(action_dim=action_dim, state_dim= state_dim)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# online actor and target actor
#actor = ActorNetwork(action_dim=action_dim, state_dim= state_dim)
#memory = Memory(1000000, state_dim, 1, 64)

desired_state = np.array([0, 0, 0, 0, 0])
desired_mask = np.array([1, 0, 0, 0, 0])
reward_history =[]


eps =1000
rollangle =[]
steerangle =[]
actions = []
time1 =[]
t0 = time.time()
state = env.reset()

prev_error_s = 0
integral_s = 0
derivative_s = 0

prev_error_r = 0
integral_r = 0
derivative_r = 0

points = 0
pid2 = 0
t_prev = t0
for i in range(eps):
    #env.render()
    time.sleep(0.01)
    Ps, Is, Ds =0.1 , 0, 0
    t11 = time.time()
    t1 = t11-t0
    dt = t1 - t_prev

    t_prev = t1
    error_s = pid2 - state[0]

    integral_s += error_s * dt
    derivative_s = (error_s - prev_error_s) / dt
    prev_error_s = error_s

    pid1 = Ps * error_s + Is * integral_s + Ds * derivative_s

    if i % 2 == 0:
        Pr, Ir, Dr = 1,0,0.5

        error_r = desired_state[2]-state[2]

        integral_r += error_r * dt
        derivative_r = (error_r - prev_error_r) / dt
        prev_error_r = error_r

        pid2 = Pr * error_r + Ir * integral_r + Dr * derivative_r

    action = pid1
   # print(action)
    #action = np.round(action).astype(np.int32)


    print (action)

    state, reward, done, info = env.step(action.flatten())

    print (state)


    points = points + reward
    #print("reward_history length: ", len(reward_history))
    print ("end")
    rollangle.append(state[2])
    steerangle.append(state[0])
    actions.append(action)
    time1.append(t1)


plt.plot(time1,rollangle,'b-')
plt.plot(time1,steerangle,'y-')
plt.plot(time1,actions,'g-')
plt.draw()
plt.pause(0.001)
#reward_history.append(points)

plt.show()
#plt.plot([i+1 for i in range(0, eps)], reward_history)
#plt.savefig("PID_plot.png")
#plt.show()

    #plt.draw()
    #plt.pause(0.001)


env.close()

