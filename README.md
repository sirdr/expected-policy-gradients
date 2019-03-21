# Deep Numerical Expected Policy Gradients

To run DDPG on the `InvertedPendulum-v1` openAI gym environment using the model outlined in [this paper](https://arxiv.org/pdf/1802.09477.pdf) simply use:

```
python ddpg.py --env_name pendulum
```

The `--learn_std` flag uses a one-layer neural network to learn the standard deviation of the output actions given the state. Without this flag, the agent learns a single standard deviation for all states.


## Running Evaluation

```
python epg.py --env_name pendulum --quadrature trapz --eval_from_checkpoint --runs 5 --learn_std
```


## Running on the Cloud

As the batch size for DNEPG especially is somewhat large, we recommend training DNEPG using a GPU. We did this using an Amazon Web Services EC2 instance initialized with the Ubuntu Deep Learning AMI and a `p2.xlarge` GPU.


To get files from the EC2 instance:
```
scp -i ~/aws/aws.pem -r {user}@{ec2-address}:~/expected-policy-gradients/results/ ./
```
