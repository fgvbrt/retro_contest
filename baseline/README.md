To run PPO training:

1) [Install docker](https://docs.docker.com/install/)

2) [Install nvidia docker](https://github.com/NVIDIA/nvidia-docker)

3) Get roms with games and place it in roms dir. See links below with information how to get Sonic roms:

    https://contest.openai.com/details

    https://github.com/openai/retro#roms

4) build image:
    
       $ docker build -t retro-ppo  -f ppo2.docker .

5) Run training:

        $ docker run --runtime=nvidia retro-ppo
