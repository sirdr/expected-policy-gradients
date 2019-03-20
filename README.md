# Expected Policy Gradients

To run DDPG on the `InvertedPendulum-v1` openAI gym environment using the model outlined in [this paper](https://arxiv.org/pdf/1802.09477.pdf) simply use:

```
python ddpg.py --env_name pendulum
```

Notes for myself:

to get files from ec2:

```
scp -i ~/aws/aws.pem -r ubuntu@ec2-52-40-19-142.us-west-2.compute.amazonaws.com:~/expected-policy-gradients/results/ ./
```
