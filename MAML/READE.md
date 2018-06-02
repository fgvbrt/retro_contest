# Setup

Setup environment with script setup.sh (you need conda for it otherwise change it)

# Run training

Meta algo uses Pyro4 for distributed training, so you should read something about 
it for example [here](https://pythonhosted.org/Pyro4/tutorials.html)


Below is basic example how to run it locally

1) start parameter server locally
    
        $ pyro4-ns &

2) start workers (for example 6 workers locally) 

        $ ./start_workers.sh localhost 9000 6 localhost

3) start meta learner

        $ python meta_learner.py --config config.yaml config_train.yaml
