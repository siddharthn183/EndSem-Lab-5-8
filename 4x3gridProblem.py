import numpy as np
import random

SMALL_ENOUGH = 0.05
GAMMA = 1     
NOISE = 0.1  

#Define all states
all_states=[]
for i in range(3):
    for j in range(4):
            all_states.append((i,j))

rewards = {}
for i in all_states:
    if i == (1,3):
        rewards[i] = -1
    elif i == (2,3):
        rewards[i] = 1
    else:
        rewards[i] = -0.04

actions = {
    (0,0):('D', 'R'), 
    (0,1):('D', 'R', 'L'),    
    (0,2):('D', 'L', 'R'),
    (0,3):('D', 'L'),
    (1, 2) : ('U', 'D', 'L', 'R'),
    (1,0):('D', 'U', 'R'),
    (2,0):('U', 'R'),
    (2,1):('U', 'L', 'R'),
    (2,2) : ('U', 'L', 'R'),
    }

policy={}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])

V={}
for s in all_states:
    if s in actions.keys():
        V[s] = 0
    if s == (1,3):
        V[s]=-1
    if s == (2,3):
        V[s]=1
    if s == (1, 1):
        V[s] = 0

iteration = 0
while True:
    biggest_change = 0
    for s in all_states:            
        if s in policy:
            
            old_v = V[s]
            new_v = 0
            
            for a in actions[s]:
                if a == 'U':
                    nxt = [s[0]-1, s[1]]
                if a == 'D':
                    nxt = [s[0]+1, s[1]]
                if a == 'L':
                    nxt = [s[0], s[1]-1]
                if a == 'R':
                    nxt = [s[0], s[1]+1]

                random_1=np.random.choice([i for i in actions[s] if i != a])
                num = random.randint(0, 100)
                if num > 80 and num <= 90:
                    if random_1 == 'U':
                        random_1 = 'L'
                    if random_1 == 'D':
                        random_1 = 'R'
                    if random_1 == 'L':
                        random_1 = 'D'
                    if random_1 == 'R':
                        random_1 = 'U'
                if num > 90:
                    if random_1 == 'U':
                        random_1 = 'R'
                    if random_1 == 'D':
                        random_1 = 'L'
                    if random_1 == 'L':
                        random_1 = 'U'
                    if random_1 == 'R':
                        random_1 = 'D'
                if random_1 == 'U':
                    act = [s[0]-1, s[1]]
                if random_1 == 'D':
                    act = [s[0]+1, s[1]]
                if random_1 == 'L':
                    act = [s[0], s[1]-1]
                if random_1 == 'R':
                    act = [s[0], s[1]+1]

                if act[0] < 3 and act[0] >= 0 and act[1] < 4 and act[1] >= 0: 
                    nxt = tuple(nxt)
                    act = tuple(act)
                    v = rewards[s] + (GAMMA * ((1-NOISE)* V[nxt] + (NOISE * V[act]))) 
                    if v > new_v: 
                        new_v = v
                        policy[s] = a
                                
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
              
    if biggest_change < SMALL_ENOUGH:
        break
    iteration += 1

for i in V:
    print(i, V[i])