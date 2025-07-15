from gym.envs.registration import register

register(
    id='KnightTour-v1',
    entry_point='gym_environments.envs:KnightTourEnv',
)