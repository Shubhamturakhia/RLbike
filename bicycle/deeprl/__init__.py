from gym.envs.registration import register

register(
    id='BicycleBalance-v0',
    entry_point='home.bike.RL_bike.Rlbike.bicycle.deeprl.envs:BicycleBalanceEnv')
