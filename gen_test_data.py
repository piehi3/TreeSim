from collections import Counter
from statistics import median, mean
import numpy as np
import random
import tree_sim_python as ts
import time

LR = 1e-3
goal_steps = 6000
score_requirement = 3000#this time in terms of time
initial_games = 5
max_power_required=5.0
INIT_TREE_POWER = 10.0

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
    print(training_data)
    training_data_save = np.array(training_data)
    np.save("score_"+str(score_requirement)+"_saved.npy", training_data_save)
 
    return training_data

start = time.time()

generate_population(None)

print("Time elapsed per sim (sec)", (time.time()-start)/initial_games)