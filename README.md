# Deep Numerical Expected Policy Gradients

To run DDPG on the `InvertedPendulum-v1` openAI gym environment using the model outlined in [this paper](https://arxiv.org/pdf/1802.09477.pdf) simply use:

```
python ddpg.py --env_name pendulum
```

The `--learn_std` flag uses a one-layer neural network to learn the standard deviation of the output actions given the state. Without this flag, the agent learns a single standard deviation for all states.


## Running Evaluation

This command should only be used locally (or if you can figure out how to using the `Moniter` environment type on an AWS instance). In any case, running this code will evaluate the learned policy from the first training run of the given algorithm and record and save a video of its performance. If you have multiple training runs (i.e. you used `--runs [number of runs]` during training where `number of runs` is greater than 1) then simply change the run number for the number of runs you want to evaluate.

```
python epg.py --env_name pendulum --quadrature trapz --eval_from_checkpoint --runs 1 --learn_std
```

## Getting Graphs and Summary Statistics


```
python get_plots_and_statistics.py --env_name pendulum
```

## Running on the Cloud

As the batch size for DNEPG especially is somewhat large, we recommend training DNEPG using a GPU. We did this using an Amazon Web Services EC2 instance initialized with the Ubuntu Deep Learning AMI and a `p2.xlarge` GPU.


To get files from the EC2 instance:
```
scp -i ~/aws/aws.pem -r {user}@{ec2-address}:~/expected-policy-gradients/results/ ./
```
