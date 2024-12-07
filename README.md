# Deep Numerical Expected Policy Gradients

Final Project for Stanford's CS 234 (See Poster Below)

![image](./cs234_poster.jpg)

***Notes & Credit:*** *Some files and functions have been adapted or borrowed from Homework 3 of the Winter 2019 version of Stanford's CS 234 taught by Emma Brunskill. I have made notes where applicable.*

To run DDPG on the `InvertedPendulum-v4` openAI gym environment for `n` episodes using the model outlined in [this paper](https://arxiv.org/pdf/1802.09477.pdf) and record a video of the resulting policy simply use:

```
python ddpg.py --env_name pendulum --output_dir <directory name> --num_episodes n --record
```

The `--learn_std` flag uses a one-layer neural network to learn the standard deviation of the output actions given the state. Without this flag, the agent learns a single standard deviation for all states.


## Running Evaluation

⚠️ ***Warning*** *`epp.py` currently breaking -- updates needed to support new version of tensorflow* 

This command should only be used locally (or if you can figure out how to using the `Moniter` environment type on an AWS instance). In any case, running this code will evaluate the learned policy from the first training run of the given algorithm and record and save a video of its performance. If you have multiple training runs (i.e. you used `--runs [number of runs]` during training where `number of runs` is greater than 1) then simply change the run number for the number of runs you want to evaluate.

```
python epg.py --env_name pendulum --quadrature trapz --eval_from_checkpoint --runs 1 --learn_std
```

Also note that if a given model was trained with the `--learn_std` flag you will need to specify it as I have done above. If the model was **not** trained with `--learn_std` you must make sure the flag is **not** used.

## Getting Graphs and Summary Statistics

If you have been collecting run results in a given `output_dir` and want to analyze and compare performance of agents, simply run the following, supplying the given directory name. `base_dir` should be the same as whichever `output_dir` your results are in.

```
python get_plots_and_statistics.py --base_dir <directory name>
```

## Running on the Cloud

As the batch size for DNEPG especially is somewhat large, we recommend training DNEPG using a GPU. We did this using an Amazon Web Services EC2 instance initialized with the Ubuntu Deep Learning AMI and a `p2.xlarge` GPU.


To get files from the EC2 instance:
```
scp -i ~/aws/aws.pem -r {user}@{ec2-address}:~/expected-policy-gradients/results/ ./
```
