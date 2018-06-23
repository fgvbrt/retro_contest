# Algorithm
Key features:
 - joint [PPO](https://arxiv.org/abs/1707.06347) training on all train games
 - mixup
 - exploration bonus to reward based on observations and x distance
 - training on test level
 - choosing best weights among several candidates during first few test episodes 


# Training
To run PPO training:

1) [Install docker](https://docs.docker.com/install/)

2) [Install nvidia docker](https://github.com/NVIDIA/nvidia-docker)

3) build image:
    
       $ docker build -t retro-ppo  -f ppo2.docker .

4) Run training:

        $ docker run --runtime=nvidia retro-ppo
