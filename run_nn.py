
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from collections import Counter
from statistics import median, mean
import numpy as np
import random
import tree_sim as ts

import matplotlib.pyplot as plt

LR = 1e-3
goal_steps = 6000
score_requirement = 3000#this time in terms of time
initial_games = 5000
max_power_required=5.0
INIT_TREE_POWER = 10.0

def model_stratagy(model,tree,cell,m,prev_observation):
    prediction = model.predict(prev_observation.reshape(-1, len(prev_observation), 1))
    return np.argmax(prediction[0])

def generate_population(model):
    # [OBS, MOVES]
    global score_requirement
 
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    print('Score Requirement:', score_requirement)
    for game_index in range(initial_games):
        print('Simulation ', game_index, " out of ", str(initial_games), '\r', end='')
        # reset env to play again
        sim = ts.TreeSim()

        if not model:
            strat = ts.random_stategy
        else:
            strat = lambda tree,cell,m,prev_observation: model_stratagy(model,tree,cell,m,prev_observation)

        sim.createTree(np.array([16,16,0],dtype=int),strat,INIT_TREE_POWER)
 
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = sim.trees[0].cells[0].getObservation(sim.model)
        # for each frame in 200
        energies=[]
        values=[]
        powers=[]
        for i in range(goal_steps):
            done = not sim.step(prev_observation)
            action = sim.trees[0].last_move

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            game_memory.append([prev_observation, action])
            prev_observation = sim.trees[0].last_observation

            energies.append(sim.trees[0].energy)
            powers.append(sim.trees[0].power)
            values.append(i)

            if done:
                break
            else:
                score+=1
        
        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement and max(powers)>max_power_required:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
 
                action_sample = [0, 0, 0, 0, 0, 0]*3
                action_sample[data[1]] = 1
                output = action_sample
                # saving our training data
                training_data.append([data[0], output])

            """
            fig, axs = plt.subplots(2)
            axs[0].plot(values,energies)
            axs[1].plot(values,powers)
            plt.savefig("output/"+str(game_index)+"energy_over_time.png")
            plt.close()

            ts.drawFrame(game_index,sim)
            plt.close()
            """
 
        # save overall scores
        scores.append(score)
 
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Score Requirement:', score_requirement)
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    score_requirement = mean(accepted_scores)
 
    # just in case you wanted to reference later
    training_data_save = np.array([training_data, score_requirement])
    np.save('saved.npy', training_data_save)
 
    return training_data


def create_dummy_model(training_data):
    shape_second_parameter = len(training_data[0][0])
    x = np.array([i[0] for i in training_data])
    X = x.reshape(-1, shape_second_parameter, 1)
    y = [i[1] for i in training_data]
    model = create_neural_network_model(input_size=len(X[0]), output_size=len(y[0]))
    return model
 
def create_neural_network_model(input_size, output_size):
    network = input_data(shape=[None, input_size, 1], name='input')
    network = tflearn.fully_connected(network, 32)
    network = tflearn.fully_connected(network, 32)
    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, name='targets')
    model = tflearn.DNN(network, tensorboard_dir='tflearn_logs')
 
    return model

def train_model(training_data, model=False):
    shape_second_parameter = len(training_data[0][0])
    x = np.array([i[0] for i in training_data])
    X = x.reshape(-1, shape_second_parameter, 1)
    y = [i[1] for i in training_data]
 
    model.fit({'input': X}, {'targets': y}, n_epoch=10, batch_size=16, show_metric=True)
    model.save('tree_trained.tflearn')
 
    return model

def evaluate(model):
    # now it's time to evaluate the trained model
    scores = []
    choices = []
    for each_game in range(20):
        score = 0
        game_memory = []
        sim = ts.TreeSim()
        strat = lambda tree,cell,m,prev_observation: model_stratagy(model,tree,cell,m,prev_observation)
        sim.createTree(np.array([16,16,0],dtype=int),strat,10.0)
        prev_observation = sim.trees[0].cells[0].getObservation(sim.model)
        for _ in range(goal_steps):
            #env.render()
 
            
            done = not sim.step(prev_prev_observation)
            action = sim.trees[0].last_move
 
            choices.append(action)
 
            prev_prev_observation = sim.trees[0].last_observation
            game_memory.append([prev_prev_observation, action])
            
            if done:
                break
            else:
                score+=1
 
        scores.append(score)
    print('Average Score is')
    print('Average Score:', sum(scores) / len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
    print('Score Requirement:', score_requirement)


if __name__ == "__main__":
 
    #some_random_games_first()
    training_data = generate_population(None)
    np.savetxt("training_data.txt",np.array(training_data))
    model = create_dummy_model(training_data)
    model = train_model(training_data, model)
    # evaluating
    evaluate(model)

"""

# recursive learning
generation = 1
while True:
    generation += 1
 
    print('Generation: ', generation)
    # training_data = initial_population(model)
    training_data = np.append(training_data, generate_population(None), axis=0)
    print('generation: ', generation, ' initial population: ', len(training_data))
    if len(training_data) == 0:
        break
    model = train_model(training_data, model)
    evaluate(model)

"""