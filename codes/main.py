# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf       # Deep Learning library
import numpy as np            # Handle matrices
from vizdoom import *         # Doom Environment
import random                 # Handling random number generation
import time                   # Handling time calculation
#from skimage import transform # Help us to preprocess the frames
from collections import deque # Ordered collection with ends
import matplotlib.pyplot as plt
import cv2

from utils import *
from network import *

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

config_path="C:/Users/Kartik/Anaconda3/envs/tf/Lib/site-packages/vizdoom/scenarios/simpler_basic.cfg"


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""

'''
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability

#main
game, possible_actions = create_environment()

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 1000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Instantiate memory
memory = Memory(max_size = memory_size)

# Render the environment
game.new_episode()
train_scores,test_scores,train_loss=[],[],[]

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    # Random action
    action = random.choice(possible_actions)
    
    # Get the rewards
    reward = game.make_action(action)
    
    # Look if the episode is finished
    done = game.is_episode_finished()
    
    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Start a new episode
        game.new_episode()
        
        # First we need a state
        state = game.get_state().screen_buffer
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Our state is now the next_state
        state = next_state
        
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

# Saver will help us to save our model
saver = tf.train.Saver()
'''
if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0

        # Init the game
        game.init()

        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            #print("episode :",episode)
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1
                
                # Increase decay_step
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # Do the action
                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))
                    train_scores.append(total_reward)
                    train_loss.append(loss)
                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                 # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model_1000.ckpt")
                print("Model Saved")

    test_scores=[]    
    for i in range(20):                
        with tf.Session() as sess:
            new_saver=tf.train.import_meta_graph('model.ckpt.meta')
            game, possible_actions = create_environment()
            
            totalScore = 0
            
            # Load the model
            new_saver.restore(sess, "model.ckpt")
            sess.run(tf.global_variables_initializer())
            game.init()
            
            for i in range(1):
                
                done = False
                
                game.new_episode()
                
                state = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, state, True)
                    
                while not game.is_episode_finished():
                    # Take the biggest Q value (= the best action)
                    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
                    
                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)
                    action = possible_actions[int(choice)]
                    
                    game.make_action(action)
                    done = game.is_episode_finished()
                    score = game.get_total_reward()
                    #print("SCORE : ",score )
                    time.sleep(0.1)
                    if done:
                        break  
                        
                    else:
                        #print("else")
                        next_state = game.get_state().screen_buffer
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        state = next_state
                        
                score = game.get_total_reward()
                test_scores.append(score)
                print("Score : ", score)
            game.close()

with tf.Session() as sess:
    
    game, possible_actions = create_environment()
    
    totalScore = 0
    
    # Load the model
    saver.restore(sess, "./models/model_1000.ckpt")
    game.init()
    for i in range(50):
        
        done = False
        
        game.new_episode()
        
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            time.sleep(0.1)
            if done:
                break  
                
            else:
                #print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
                
        score = game.get_total_reward()
        test_scores.append(score)
        print("Score: ", score)
    game.close()
    

plt.title("Training episodes vs average loss per 50 eps")
plt.xlabel("Training episodes")
plt.ylabel("avg loss per 50 eps")
plt.plot(ep,loss_per_50[1:],label="avg loss",c='r')
plt.legend(loc="upper right")
plt.show()