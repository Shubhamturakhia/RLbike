from gym.envs.registration import register

register(
    id='BicycleBalance-v0',
    entry_point='deeprl.envs:BicycleBalanceEnv')