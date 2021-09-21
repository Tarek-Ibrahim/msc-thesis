from gym.envs.registration import register

register(
    id='halfcheetah_custom-v3',
    entry_point='codes.envs.half_cheetah_custom_torch:HalfCheetahEnv',
    max_episode_steps=300,
    reward_threshold=4800.0,
)

