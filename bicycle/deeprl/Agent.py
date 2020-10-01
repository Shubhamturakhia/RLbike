# TODO: Make sure the parameters for the agent are initialized in the Initialize function
# TODO: Make sure the updation of actor anfd critic target networks are done
"""
  TODO: 1. Function to get sample buffer
        2. Function to do the Learning process
        3. Function
"""
from Resources import OUNoise, ReplayBuffer
from Actor_Network import *
from Critic_Network import *


class Agent:

    def __init__(self, alpha=0.0001, beta=0.001, tau=0.001, env=None, input_dim=[8], gamma=0.99, n_act=1,
                 max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.RB = ReplayBuffer(max_size, input_dim, n_act)
        self.n_act =n_act
        self.batch_size = batch_size
        self.sess = tf.compat.v1.Session()
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.noise = OUNoise(mu=np.zeros(n_act))
        print("ACTOR NETWORK AGENT")
        self.actor = ActorNN(learning_rate=alpha, input_dim=input_dim,
                             name='Actor',
                             sess=self.sess, n_act=n_act,
                             layer1_dims=layer1_size, layer2_dims=layer2_size, action_bound=env.action_space.high)
        print("CRITIC NETWORK AGENT")
        self.critic = CriticNN(learning_rate=beta, input_dim=input_dim,
                               name='Critic', sess=self.sess, n_act=n_act,
                               layer1_dims=layer1_size, layer2_dims=layer2_size)

        self.target_a = ActorNN( learning_rate=alpha, n_act=n_act,
                                name='TargetActor',
                                input_dim=input_dim, sess=self.sess, layer1_dims=layer1_size,
                                layer2_dims=layer2_size, action_bound=env.action_space.high)

        self.target_c = CriticNN(learning_rate=beta, n_act=n_act,
                                 name='TargetCritic', input_dim=input_dim,
                                 sess=self.sess, layer1_dims=layer1_size, layer2_dims=layer2_size)

        self.update_critic = [self.target_c.network_parameters[i].assign(
            tf.multiply(self.critic.network_parameters[i], self.tau) +
            tf.multiply(self.target_c.network_parameters[i], 1. - self.tau))
            for i in range(len(self.target_c.network_parameters))]

        self.update_actor = [self.target_a.network_parameters[i].assign(
            tf.multiply(self.actor.network_parameters[i], self.tau) +
            tf.multiply(self.target_a.network_parameters[i], 1. - self.tau))
            for i in range(len(self.target_a.network_parameters))]

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_a.compile(optimizer=Adam(learning_rate=alpha))
        self.target_c.compile(optimizer=Adam(learning_rate=beta))
        #self.update_parameters(initial_session=True)
        self.update_parameters(tau=1)

    def learning_stage(self):
        if self.RB.memory_cntr < self.batch_size:
            return

        state, action, reward, new_state, flag_complete = self.RB.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_a(new_states)
            critic_value_ = tf.squeeze(self.target_c(new_state, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - flag_complete)
            critic_loss = tf.compat.v1.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_parameters()

    def get_sample_buffer(self, state, action, reward, new_state, flag_complete):
        #print(flag_complete)
        return self.RB.transition(state, action, reward, new_state, flag_complete)

    def update_parameters(self, tau=None):

        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_a.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_a.set_weights(weights)

        weights = []
        targets = self.target_c.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_c.set_weights(weights)

    def action(self, observation, evaluate=False):
        state = tf.convert_to_tensor(np.array([observation], dtype=np.float32))
        #state = tf.reshape(state, shape=[5,-1])
        #state = observation[np.newaxis:1]
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_act], mean=0.0, stddev=0.1)
        # note that if the environment has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]