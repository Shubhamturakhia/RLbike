from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from agents.BaseAgent import BaseAgent
from utils.utilities import *
import os
from noises.OUNoise import *
from gym.wrappers import Monitor
from gym.spaces import *
from envs.env_wrapper import *
from mems.replay import *
from nets.networks import *
import serial
from struct import *

# DDPG algorithm for bike
class Agent_DDPG(BaseAgent):
    """
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess,
                             layer1_size, layer2_size)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor',
                                  input_dims, self.sess, layer1_size,
                                  layer2_size, env.action_space.high)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size)
    """
    def __init__(self, sess, actor, critic, memory, env, env_name, max_episode=3000,
                 b4_train=5000, plot=True,
                 max_test_epoch=1000, render=True, save_interval=10,
                 save_plot_interval=50,
                 OUnoise_theta = 0.15,
                 OUnoise_sigma = 0.2,
                 max_step_per_game = 10000):

        super(Agent_DDPG, self).__init__(sess, env=env, render=render, max_episode=max_episode,
                                   env_name=env_name, warm_up=b4_train,save_plot_interval=save_plot_interval,
                                   max_test_epoch=max_test_epoch, save_interval=save_interval,
                                   max_step_per_game = max_step_per_game)

        # Assign parameter to the variables of the class
        self.critic = critic
        self.actor = actor
        self.memory = memory
        self.batch_size = self.memory.batch_size
        self.plot = plot

        # Initialize the exploration noise
        self.noise = OUNoise(self.actor.action_dim, theta=OUnoise_theta, sigma=OUnoise_sigma)

        np.random.seed(123)
        tf.set_random_seed(123)
        if type(self.env) == ContinuousWrapper:
            self.env.seed(123)

        # Initialize the whole network policy
        self.sess.run(tf.global_variables_initializer())

    def restoreModel(self):
        self.restore()

    def bike_learning(self):

        print("Start learning process....")
        episodes = 0

        # Initialize the variables for saving the trajectories
        if self.plot:
            total_reward = []
            total_step = []
            if "BicycleBalance-v0" in self.env_name:
                back_trajectory = []
                front_trajectory = []
                term1 = []
                term2 = []
                term3 = []

        # iterate until reach the maximum step
        while episodes < self.max_episode:

            # re-initialize per episode variables
            current_state = self.env.reset()
            per_game_step = 0
            per_game_reward = 0
            per_game_q = 0
            done = False

            current_state = current_state[np.newaxis]
            # re-initialize the noise process
            self.noise.reset()

            # iterate through the whole episode
            while not done and per_game_step < self.max_step_per_game:

                # simulate the environment
                if self.render:
                    self.env.render()

                # Take a random action in the warm-up stage
                if self.warm_up > self.memory.get_curr_size():
                    # Random action which would be stored in the sample buffer
                    # After that in sample buffer can be retrieved
                    action = self.env.action_space.sample()
                else:
                    # Action exploration

                    # state = state[np.newaxis, :]
                    # mu = self.actor.predict(state)  # returns list of list
                    # noise = self.noise()
                    # mu_prime = mu + noise

                    #noise = ((self.max_episode-episodes)/self.max_episode)*self.noise.noise()
                    noise = self.noise.noise()
                    pure_action = self.action(current_state)
                    action = pure_action + noise
                    action = action.flatten()

                # evaluate the Q(s,a) value
                reshaped_action = action.reshape(1, -1)
                per_game_q += self.sess.run(self.critic.network,
                                            feed_dict={self.critic.action: reshaped_action,
                                                       self.critic.state: current_state})

                # This goes to the Env and calculates the reward for each step
                # also calculate the next_state, done boolean value and info
                next_state, reward, done, info = self.env.step(action)

                next_state = next_state[np.newaxis]

                terminal = 0 \
                    if done \
                    else 1

                # store experiences
                self.memory.add([current_state, next_state, reward, terminal, action], reward)

                # train the networks
                if self.warm_up < self.memory.get_curr_size():
                    if self.batch_size < self.memory.get_curr_size():
                        # random batch_size samples
                        samples, idxs = self.memory.sample()
                        s = samples[0]
                        next_s = samples[1]
                        r = samples[2]
                        t = samples[3]
                        a = samples[4]
                        a = a.reshape(self.batch_size, -1)


                        target_action = self.actor_target_predict(next_s)

                        y = self.critic_target_predict(next_s, target_action)
                        y = r + t * self.gamma * y
                        _, loss, _ = self.update_critic(y, a, s)
                        actor_action = self.action(s)

                        a_grads = self.get_action_gradient(s, actor_action)
                        self.update_actor(s, a_grads[0])

                        self.sess.run([self.actor.update_op, self.critic.update_op])

                current_state = next_state
                per_game_step += 1

                per_game_reward += reward

            if self.plot:
                total_reward.append(per_game_reward)
                #total_step.append(per_game_step)
                if "BicycleBalance-v0" in self.env_name:
                    back_trajectory.append([self.env.env.get_xbhist(), self.env.env.get_ybhist()])
                    front_trajectory.append([self.env.env.get_xfhist(), self.env.env.get_yfhist()])
                    #terms = self.env.env.getReward()
                    term1.append(self.env.env.getReward()[0])
                    term2.append(self.env.env.getReward()[1])
                    term3.append(self.env.env.getReward()[2])
                    # goals.append(self.env.env.get_goal())

            # Save model
            if episodes % self.save_interval == 0 and episodes != 0:
                self.save()

            if self.plot and episodes % self.save_plot_interval == 0:
                self.save_plot_data("total_step", np.asarray(total_step), is_train=True)
                self.save_plot_data("total_reward", np.asarray(total_reward), is_train=True)
                if "Bicycle" in self.env_name:
                    if "Bicycle-v0" in self.env_name:
                        self.save_plot_data("back_trajectory", np.asarray([back_trajectory, [None, 1]]),
                                            is_train=True)
                        self.save_plot_data("front_trajectory", np.asarray([front_trajectory, [None, 1]]),
                                            is_train=True)
                    else:
                        #self.save_plot_data("back_trajectory", np.asarray([back_trajectory,
                        # [np.array([self.env.env.x_goal,self.env.env.y_goal]), 1]]), is_train=True)
                        #self.save_plot_data("front_trajectory", np.asarray([front_trajectory,
                        # [np.array([self.env.env.x_goal,self.env.env.y_goal]), 1]]), is_train=True)
                        self.save_plot_data("term1", np.asarray(term1), is_train=True)
                        self.save_plot_data("term2", np.asarray(term2), is_train=True)
                        self.save_plot_data("term3", np.asarray(term3), is_train=True)

                        # self.save_plot_data("goal", np.asarray(goals), is_train=True)

            episodes += 1
            per_game_q = per_game_q/per_game_step

            print("Episode: {}/{} | Reward: {} | Q-value: {} | Steps: {}".format(episodes,self.max_episode,
                                                                          per_game_reward,
                                                                          per_game_q,
                                                                          per_game_step))

        if self.plot:
            #self.save_plot_data("total_step", np.asarray(total_step),is_train=True)
            self.save_plot_data("total_reward", np.asarray(total_reward),is_train=True)
            self.save_plot_data("term1", np.asarray(term1), is_train=True)
            self.save_plot_data("term2", np.asarray(term2), is_train=True)
            self.save_plot_data("term3", np.asarray(term3), is_train=True)



    def evaluate(self):
        print("Start Evaluating...")

        total_reward = 0
        # Restore the policy from file
        self.restore()

        for epoch in range(self.max_test_epoch):
            print("Episode: %s" % (epoch))
            # start the environment
            state = self.env.reset()
            #print(state)
            step = 0
            done = False
            total_reward = 0

            # start one episode
            while not done and step < self.max_step_per_game:
                if (self.render):
                    self.env.render()

                reshaped_state = state[np.newaxis]
                #print(reshaped_state)

                # predict action using the network policy
                action = self.action(reshaped_state)
                #print(action)

                # get the next state, reward and terminate signal
                state, reward, done, _ = self.env.step(action.flatten())
                total_reward += reward
                step += 1

            print("Episode: {}/{} | Reward: {}" .format (epoch, self.max_test_epoch,total_reward))

    def hardware_(self):
        print ("Hardware setup testing....")

        self.restore()

        #TODO: Get the state of the bike at instantaneous time from Psoc (state)
        #TODO: Return the action from the jetson to the Psoc
        #TODO: Get the new state and apply it the the state var (used previously)
        with serial.Serial("/dev/ttyTHS2", baudrate=9600) as ser:
            while True:
                state = ser.readline()  # state from bike
                valid = verify_checksum(state)  # Check the validity of the data

                if valid:
                    parsed_state = unpack('<fffffxx', state)
                    parsed_state = np.array(parsed_state)
                    print(parsed_state)
                    reshaped_state = parsed_state[np.newaxis]
                    print(reshaped_state)

                    action = self.action(reshaped_state)
                    action =np.array(action)

                    print (action)
                    packing_action = pack('!iBc', int(action[0][0]), 0, b'\n')
                    packing_action = pack('!iBc', int(action[0][0]), generate_checksum(packing_action), b'\n')
                    ser.write(packing_action)


    '''
    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    '''
    def critic_target_predict(self, state, action):
        return self.sess.run(self.critic.target_network,
                             feed_dict={self.critic.target_action: action, self.critic.target_state: state})

    def action(self, state):
        return (self.sess.run(self.actor.network,
                              feed_dict={self.actor.state: state}))

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, 'DDpg_checkpoints')

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, 'DDpg_checkpoints')

    def actor_target_predict(self, state):
        return self.sess.run(self.actor.target_network,
                             feed_dict={self.actor.target_state: state})

    def update_critic(self, y, a, s):
        return self.sess.run([self.critic.mean_loss, self.critic.loss, self.critic.train],
                             feed_dict={self.critic.y: y, self.critic.action: a, self.critic.state: s})

    def update_actor(self, s, a_grads):
        self.sess.run(self.actor.train,
                      feed_dict= {self.actor.state: s, self.actor.action_gradient: a_grads})

    def get_action_gradient(self, s, a):
        return self.sess.run(self.critic.action_gradient,
                             feed_dict= {self.critic.state: s, self.critic.action: a})

def verify_checksum (data):
    if len(data)==(4*5+2):
        return (np.sum(list(data)[:-2])% 255) == int (data[-2])
    print("length", len(data))
    return False

def generate_checksum (data):
    return (np.sum(list(data)[:-2])% 255)

