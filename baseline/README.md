To run PPO training:

1) [Install docker](https://docs.docker.com/install/)

2) [Install nvidia docker](https://github.com/NVIDIA/nvidia-docker)

3) build image:
    
       $ docker build -t retro-ppo  -f ppo2.docker .

4) Run training:

        $ docker run --runtime=nvidia retro-ppo
