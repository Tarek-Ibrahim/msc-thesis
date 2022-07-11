import gym
import gym_custom
import mujoco_py

env=gym.make('Ant-v2')
env2=gym.make('halfcheetah_custom_rand-v2')

s=env2.reset()

print(s)
