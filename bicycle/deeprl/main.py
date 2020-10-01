# TODO: Defining the environment and executing the algorithm steps
# TODO: Game Engine Model and its required installation
# TODO: Game Env with execution of algorithm

import gym
import matplotlib.pyplot as plt
from Agent import *
from absl import logging,flags,app
from absl.flags import FLAGS
import time
from envs.env_wrapper import *
from BicycleRender import BicycleRender

def main(_argv):

    tn1 = time.time()
    # Environment using OpenAI gym
    env = ContinuousWrapper(gym.make('BicycleBalance-v0'))
    # env.reset()
    Episodes=100

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    # Initialize agent
    Ag_controller= Agent(input_dim=state_dim,
                          env=env,
                         n_act=action_dim)
    logging.info("Agent initialized....")
    """
    alpha = 10e-4
    beta = 10e-3
    input_dims = [3]
    tau = 0.001
    gamma = 0.99
    n_act = 1,
    max_size = 10e6,
    layer1_size = 400
    layer2_size = 300
    """"""
    np.random.seed(123)
    tf.compat.v1.set_random_seed(123)
    if type(env) == ContinuousWrapper:
        env.seed(123)
    """
    reward_history =[]

    for i in range(Episodes):
        logging.info("Starting episode run...")

        flag_complete = False
        observation = env.reset()
        points = 0
        evaluate = False

        while not flag_complete:
            t1 = time.time()
            # This follow the steps exactly in the DDPG algorithm
            # Action chosen for the initial stage and then the parameters (SARS') are returned to be stored in buffer
            action_chosen = Ag_controller.action(observation, evaluate)
            new_state, reward, flag_complete, info = env.step(action_chosen)
            #print (new_state, reward, flag_complete)
            Ag_controller.get_sample_buffer(observation, action_chosen, reward, new_state, flag_complete)

            # Learn stage
            Ag_controller.learning_stage()

            # Reward points calculation
            observation = new_state
            points = points + reward
            env.render()
            t2 = time.time()
        reward_history.append(points)
        print("Episode No : {}/{} | Reward: {} | Mean Score for 10 episodes: {}".
              format(i, Episodes, float(points), np.mean(reward_history[-10:])))
        tn2 = time.time()

    plt.plot([i + 1 for i in range(0, Episodes)], reward_history)
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
