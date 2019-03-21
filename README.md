# Expected Policy Gradients

To run DDPG on the `InvertedPendulum-v1` openAI gym environment using the model outlined in [this paper](https://arxiv.org/pdf/1802.09477.pdf) simply use:

```
python ddpg.py --env_name pendulum
```

The `--learn_std` flag uses a one-layer neural network to learn the standard deviation of the output actions given the state. Without this flag, the agent learns a single standard deviation for all states.


## Running Evaluation

```
python epg.py --env_name pendulum --quadrature trapz --eval_from_checkpoint --runs 5 --learn_std
```

Notes for myself:

to get files from ec2:

```
scp -i ~/aws/aws.pem -r ubuntu@ec2-52-40-19-142.us-west-2.compute.amazonaws.com:~/expected-policy-gradients/results/ ./
```
