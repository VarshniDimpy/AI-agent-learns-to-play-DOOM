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

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

config_path="C:/Users/Kartik/Anaconda3/envs/tf/Lib/site-packages/vizdoom/scenarios/simpler_basic.cfg"




def create_environment():
    game = DoomGame()
    
    # Load the correct configuration
    game.load_config("C:/Users/Kartik/Anaconda3/envs/tf/Lib/site-packages/vizdoom/scenarios/basic.cfg")
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("C:/Users/Kartik/Anaconda3/envs/tf/Lib/site-packages/vizdoom/scenarios/basic.wad")
    game.set_screen_format(ScreenFormat.GRAY8)
    #game.set_screen_resolution(ScreenResolution.RES_1024X576)
        
    game.init()
    
    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    
    return game, possible_actions
       

def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()
    


def preprocess_frame(state_frame):
    
    #print("shape:",state_frame.shape)
    not_roof=state_frame[30:-10,30:-30]
    not_roof=state_frame/255.0
    #print(type(not_roof))
    #print(not_roof.shape)
    new_frame=cv2.resize(not_roof,(84,84))
    #print(new_frame.shape)
    return new_frame

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    #print("state = ",state)
    #print("state = ",state.shape)
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames
