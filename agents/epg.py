import os
import argparse
import sys
import logging
import time
import numpy as np
from scipy import integrate
from copy import copy

import tensorflow as tf

import gym
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import get_config

from .base_pg import PG, Model

from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])

# class QuadraticCritic(Model):
#     def __init__(self, name='quad_critic'):
#         super().__init__(name=name)

#     def A(self, obs):
#         output = None
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#             #h = tf.layers.batch_normalization(obs, training=training)
#             h = tf.layers.dense(obs, 400, activation=None, kernel_initializer=tf.initializers.variance_scaling())
#             h = tf.nn.relu(h)
#             h = tf.layers.dense(h, 300, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
#             out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)
#             output = tf.layers.dense(h, 1, activation=output_activation, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
#         return output

#     def B(self, obs):
#         output = None
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#             #h = tf.layers.batch_normalization(obs, training=training)
#             h = tf.layers.dense(obs, 400, activation=None, kernel_initializer=tf.initializers.variance_scaling())
#             h = tf.nn.relu(h)
#             h = tf.layers.dense(h, 300, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
#             out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)
#             output = tf.layers.dense(h, 1, activation=output_activation, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
#         return output

#     def __call__(self, 
#                 obs, 
#                 actions,
#                 output_activation=None):

#         output = None
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#             A_s = self.A(obs)
#             B_s = self.B(obs)
#             output = actions*A_s*actions + B_s*actions
#         return output

class Actor(Model):
    def __init__(self, output_dim, name='actor', discrete = False, max_action=1, learn_std=False):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.max_action = max_action
        self.discrete = discrete
        self.learn_std = learn_std

    def __call__(self,
                obs,
                output_activation=None,
                reuse=False,
                training=True):

        output = None
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            #h = tf.layers.batch_normalization(obs, training=training)
            h = tf.layers.dense(obs, 400, activation=None, kernel_initializer=tf.initializers.variance_scaling())
            h = tf.nn.relu(h)
            h = tf.layers.dense(h, 300, activation=None, kernel_initializer=tf.initializers.variance_scaling())
            h = tf.nn.relu(h)
            out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)

            if not self.discrete:
                #output_logstd = tf.layers.dense(obs, self.num_actions, activation=None, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
                #return [self.max_action*output, output_logstd]
                #return self.max_action*output
                if not self.learn_std:
                    h = tf.layers.dense(h, self.output_dim, activation=None, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
                    return self.max_action*tf.nn.tanh(h)
                else:
                    h = tf.layers.dense(h, 2*self.output_dim, activation=None, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
                    #h2 = tf.layers.dense(obs, self.output_dim, activation=None, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
                    h1 = h[:, : self.output_dim]
                    h2 = h[:, self.output_dim :]
                return self.max_action*tf.nn.tanh(h1), h2
            else:
                h = tf.layers.dense(h, self.output_dim, activation=None, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
                output = tf.nn.softmax(h)
                return output


class Critic(Model):
    def __init__(self, name='critic', batch_norm=False):
        super().__init__(name=name)
        self.batch_norm = batch_norm

    def __call__(self, 
                obs, 
                actions,
                output_activation=None,
                reuse=False,
                training=True):

        output = None
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            #h = tf.layers.batch_normalization(obs, training=training)
            obs_act = tf.concat([obs, tf.cast(actions, tf.float32)], 1) 
            h = tf.layers.dense(obs_act, 400, activation=None, kernel_initializer=tf.initializers.variance_scaling())
            h = tf.nn.relu(h)
            h = tf.layers.dense(h, 300, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
            out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)
            output = tf.layers.dense(h, 1, activation=output_activation, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
        return output

class EPG(PG):
    """
    Class for Expected Policy Gradients, Inherets from the generic Policy Gradient class
    """
    def __init__(self, env, config, actor=None, critic=None, quadrature = 'riemann', logger=None, num_actions=1000, run=0, learn_std=False):

        super().__init__(env, config, run=run, logger=logger)
        self.agent_name = "epg-{}".format(quadrature)
        self.learn_std = learn_std
        if actor is None:
            if self.discrete:
                actor = Actor(self.action_dim, discrete=self.discrete)
                self.num_actions = self.action_dim # this is for discretising the action space in continuous domain
                self.quadrature = "discrete"
            else:
                actor = Actor(self.action_dim, discrete=self.discrete, max_action=self.action_high)
                self.num_actions = num_actions # number of actions to use for integral
        self.actor = actor

        if critic is None:
            critic = Critic()
        self.critic = critic

        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor

        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        self.tau = 0.005
        self.actor_lr = 0.001
        self.quadrature = quadrature
        self.build()

    def add_placeholders_op(self):
        """
        Add placeholders for observation, action, and advantage:
            self.observation_placeholder, type: tf.float32
            self.action_placeholder, type: depends on the self.discrete
            self.advantage_placeholder, type: tf.float32
        """
        self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.next_observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int32, shape=(None, 1))
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))

        self.weights_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
        self.reward_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
        self.done_placeholder = tf.placeholder(tf.bool, shape=(None, 1))

        self.training_placeholder = tf.placeholder(tf.bool)
        self.num_states_placeholder = tf.placeholder(tf.float32, shape=())

    def get_policy_from_actor_op(self, actor, obs):

        if self.discrete:
            action_logits = actor(obs)#training=self.training_placeholder)
            sampled_action = tf.multinomial(action_logits, 1)
            return action_logits, sampled_action, None
        else:
            action_output = actor(obs) #training=self.training_placeholder)
            #log_std = action_output[1]
            if not self.learn_std:
                with tf.variable_scope(actor.name, reuse=tf.AUTO_REUSE):
                    log_std = tf.get_variable("log_std", shape=(self.action_dim))
                action_means = action_output
            else:
                action_means = action_output[0]
                action_means = tf.expand_dims(action_means, axis=1) 
                log_std = action_output[1]
            shape = tf.shape(action_means)
            epsilon = tf.random_normal(shape)
            sampled_action = action_means + tf.multiply(epsilon, tf.exp(log_std))
            # TODO: clip here ? or clip only in act(), revert back to log_std ?
            return action_means, sampled_action, log_std

    def get_likelihood_op(self, target_actions, pred_actions, log_std=None):

        if self.discrete:
            logprob = -1*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(target_actions), logits=pred_actions)
            prob = tf.exp(logprob)
        else:
            logprob = tf.contrib.distributions.MultivariateNormalDiag(loc=pred_actions, scale_diag=tf.exp(log_std)).log_prob(target_actions)
            prob = tf.exp(logprob)
        return logprob, prob

    def add_actor_loss_op(self):
        self.loss_integrand = tf.multiply(tf.expand_dims(self.prob, axis=1), self.critic_output)

        loss_integrand_reshaped = tf.reshape(self.loss_integrand, [-1, self.num_actions])
        loss_integrand_reshaped_avg = (loss_integrand_reshaped[:, :-1] + loss_integrand_reshaped[:, 1:])/2.0
        loss_integrand_avg = tf.reshape(loss_integrand_reshaped_avg, [-1, 1])

        self.loss_integrand_weighted_trapz = tf.multiply(self.weights_placeholder, loss_integrand_avg)
        self.loss_integrand_weighted_riemann = tf.multiply(self.weights_placeholder, self.loss_integrand)

        self.loss_integral_trapz = tf.reduce_sum(self.loss_integrand_weighted_trapz)/self.num_states_placeholder
        self.loss_integral_riemann = tf.reduce_sum(self.loss_integrand_weighted_riemann )/self.num_states_placeholder

        if self.quadrature == "trapz":
            self.loss_integral = self.loss_integral_trapz
        else:
            self.loss_integral = self.loss_integral_riemann

        self.loss = -1*self.loss_integral

    def add_actor_optimizer_op(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        v_list = self.actor.vars
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Needed for Batch Normalization to work properly
        self.actor_grads_and_vars = opt.compute_gradients(self.loss, var_list=v_list)
        grads = [g for g,v in self.actor_grads_and_vars]
        self.grad_norm = tf.global_norm(grads)
        self.train_op = opt.apply_gradients(self.actor_grads_and_vars)
        #self.train_op = tf.group([train_op, update_ops])

    def add_critic_loss_op(self, scope="critic"):
        self.y = self.reward_placeholder + self.config.gamma*tf.multiply(tf.cast(tf.logical_not(self.done_placeholder), tf.float32), self.target_critic_output)
        self.critic_loss = tf.losses.mean_squared_error(self.y, self.critic_output,  scope=self.critic.name)

    def add_critic_optimizer_op(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        v_list = self.critic.vars
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.actor_grads_and_vars = opt.compute_gradients(self.critic_loss, var_list=v_list)
        self.train_critic_op = opt.apply_gradients(self.actor_grads_and_vars)
        #self.train_critic_op = tf.group([train_critic_op, update_ops])

    def add_soft_update_target_op(self):
        critic_v_list = self.critic.vars
        target_critic_v_list = self.target_critic.vars
        self.update_target_critic_op = tf.group(*[tf.assign(target_v, self.tau*v + (1 - self.tau)*target_v) for v, target_v in zip(critic_v_list, target_critic_v_list)])

        actor_v_list = self.actor.vars
        target_actor_v_list = self.target_actor.vars
        self.update_target_actor_op = tf.group(*[tf.assign(target_v, self.tau*v + (1 - self.tau)*target_v) for v, target_v in zip(actor_v_list, target_actor_v_list)])

    def add_init_update_target_op(self):
        critic_v_list = self.critic.vars
        target_critic_v_list = self.target_critic.vars
        self.init_target_critic_op = tf.group(*[tf.assign(target_v, v) for v, target_v in zip(critic_v_list, target_critic_v_list)])

        actor_v_list = self.actor.vars
        target_actor_v_list = self.target_actor.vars
        self.init_target_actor_op = tf.group(*[tf.assign(target_v, v) for v, target_v in zip(actor_v_list, target_actor_v_list)])

    def build(self):

        # add placeholders
        self.add_placeholders_op()

        # create stochastic policy from actor network
        self.pred_actions, self.sampled_actions, self.log_std = self.get_policy_from_actor_op(self.actor,  
                                                                                                self.observation_placeholder)
        self.target_pred_actions, self.target_sampled_actions, self.target_log_std = self.get_policy_from_actor_op(self.target_actor, 
                                                                                                                    self.next_observation_placeholder)

        self.pred_actions_next, self.sampled_actions_next, self.log_std_next = self.get_policy_from_actor_op(self.actor,  
                                                                                                self.next_observation_placeholder)

        self.logprob, self.prob = self.get_likelihood_op(self.action_placeholder, self.pred_actions, log_std = self.log_std)

        self.target_logprob, self.target_prob = self.get_likelihood_op(self.target_sampled_actions, 
                                                                        self.target_pred_actions, 
                                                                        log_std = self.target_log_std)

        # get critic values
        self.critic_output = self.critic(self.observation_placeholder, self.action_placeholder)
        self.target_critic_output = self.target_critic(self.next_observation_placeholder, self.target_sampled_actions)

        self.add_actor_loss_op()
        self.add_actor_optimizer_op()

        self.add_critic_loss_op()
        self.add_critic_optimizer_op()

        self.add_soft_update_target_op()
        self.add_init_update_target_op()

    def initialize(self):
        # create tf session
        self.sess = tf.Session()

        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)
        self.sess.run([self.init_target_actor_op, self.init_target_critic_op])

    def update_critic(self, actions, observations, next_observations, rewards, done):

        self.sess.run(self.train_critic_op, feed_dict={
                        self.observation_placeholder : observations,
                        self.next_observation_placeholder : next_observations,
                        self.action_placeholder : actions,
                        self.reward_placeholder : rewards,
                        self.done_placeholder : done})

    def update_targets(self):
        self.sess.run([self.update_target_critic_op, self.update_target_actor_op], feed_dict={})

    def act(self, observation, compute_q = False):
        feed_dict = {self.observation_placeholder: observation[None]}
        if compute_q:
            action, q = self.sess.run([self.sampled_actions, self.critic_output_given_actor_output], feed_dict=feed_dict)
        else:
            action = self.sess.run(self.sampled_actions, feed_dict=feed_dict)
            q = None
        # if not self.discrete:
        #     action = np.clip(action, self.action_low, self.action_high)
        action = action[0].item()
        #print("action {}".format(action))
        return action, q

    def evaluate_policy(self, env=None, eval_episodes=10):
        if env==None: env = self.env
        rewards = []
        eval_cummulative_timesteps = []
        eval_episode_timesteps = []

        total_timesteps = 0
        episode_timesteps = 0

        for _ in range(eval_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.act(np.array(obs))
                if not self.discrete:
                    action = np.clip(action, self.action_low, self.action_high)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                episode_timesteps += 1
                total_timesteps += 1
            eval_episode_timesteps.append(episode_timesteps)
            eval_cummulative_timesteps.append(total_timesteps)
            episode_timesteps = 0
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        print("---------------------------------------")
        print("Evaluation over {} episodes: {:04.2f} +/- {:04.2f}".format(eval_episodes, avg_reward, sigma_reward))
        print("---------------------------------------")
        return rewards, eval_cummulative_timesteps, eval_episode_timesteps

    def train_actor(self, observations, stats={}):

        if self.quadrature == "analytic":
            pass

        else:

            # Get Experience From Quadrature
            def function_to_integrate(actions, obs):
                try:
                    num_actions = actions.shape[0]
                except:
                    num_actions = 1
                    actions = np.array([actions])
                if num_actions > 1:
                    obs = np.tile(obs, (num_actions, 1))
                else:
                    obs = obs[None]
                val = self.sess.run(self.loss_integrand, feed_dict = {self.observation_placeholder: obs,
                                                                        self.action_placeholder : actions[None]})
                return val

            num_states = len(observations)
            observations = np.array(observations)

            results_from_scipy = None

            if self.discrete:
                observations = np.reshape(np.tile(observations, (self.action_dim)), (self.action_dim*num_states, -1))
                actions = np.arange(self.action_dim)
                weights = np.ones(self.action_dim)
            else:
                actions = np.linspace(self.action_low,self.action_high, num=self.num_actions)
                weights = (actions[1:]-actions[:-1])
                #actions = np.random.uniform(self.action_low,self.action_high, size=100000)
                if self.quadrature == "riemann":
                    actions = (actions[:-1]+actions[1:])/2.
                # else:
                #     results = integrate.quad(function_to_integrate, self.action_low, self.action_high, args=(observation,), full_output=1, maxp1=100)
                results_from_scipy = None
                # result_list = []
                # for observation in observations:
                #     results = integrate.quadrature(function_to_integrate, self.action_low, self.action_high, args=(observation,), vec_func=False)
                #     result_list.append(results[0])
                # results_from_scipy = np.mean(result_list)
                if stats["integral_action_values"] is None:
                    stats["integral_action_values"] = actions

                observations = np.reshape(np.tile(observations, len(actions)), (len(actions)*num_states, -1))
            
            actions = np.tile(actions, (num_states))
            weights = np.tile(weights, (num_states))
            actions = actions[:, None]
            weights = weights[:, None]

            _ , prob, loss_integral, loss_integrand, grad_norm = self.sess.run([self.train_op, self.prob, self.loss_integral, self.loss_integrand, self.grad_norm], feed_dict={
                                                self.observation_placeholder : observations,
                                                self.action_placeholder : actions,
                                                self.weights_placeholder: weights,
                                                self.num_states_placeholder: num_states})

            #print("shapes --- prob: {} | critic_output: {} | loss_integrand: {}".format(prob.shape, critic_output.shape, loss_integrand.shape))
            #print("integral --- riemann: {} | scipy : {}".format(loss_integral, results[0]))
            #print("integral --- {}: {} | scipy : {}".format(self.quadrature, loss_integral, results_from_scipy))
            # print("")

            stats["grad_norms"].append(grad_norm)
            if stats["first_integrand"] is None:
                stats["first_integrand"] = loss_integrand
            stats["last_integrand"] = loss_integrand
            stats["loss_integral"].append(loss_integral)

        return stats

    def train_critic(self, observation, action, reward, next_observation, done):
        action = np.array(action)[None]
        reward = np.array([reward])[None]
        done = np.array([done])[None]
        next_observation = next_observation[None]
        observation = observation[None]
        self.update_critic(action, observation, next_observation, reward, done)


if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    # train model
    model = EPG(env, config)

    model.run()