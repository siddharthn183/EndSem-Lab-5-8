import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import sys
class jcp:
    @staticmethod
    def max_cars():
        return 20
    
    @staticmethod
    def γ():
        return 0.9
    
    @staticmethod
    def credit_reward():
        return 10
    
    @staticmethod
    def moving_reward():
        return -2
    
    @staticmethod
    def second_parking_lot_reward():
        return -4
class poisson_:
    
    def __init__(self, lam):
        self.lam = lam
        
        e = 0.01
        self.w = 0
        state = 1
        self.vals = {}
        summer = 0
        
        while(1):
            if state == 1:
                temp = poisson.pmf(self.w, self.lam) 
                if(temp <= e):
                    self.w+=1
                else:
                    self.vals[self.w] = temp
                    summer += temp
                    self.bet = self.w+1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.bet, self.lam)
                if(temp > e):
                    self.vals[self.bet] = temp
                    summer += temp
                    self.bet+=1
                else:
                    break    
        
        
        added_val = (1-summer)/(self.bet-self.w)
        for key in self.vals:
            self.vals[key] += added_val
        
            
    def f(self, n):
        try:
            Ret_value = self.vals[n]
        except(KeyError):
            Ret_value = 0
        finally:
            return Ret_value

class location:
    
    def __init__(self, req, ret):
        self.w = req                             #value of lambda for requests
        self.bet = ret                             #value of lambda for returns
        self.poisson = poisson_(self.w)
        self.poisson = poisson_(self.bet)


A = location(3,3)
B = location(4,2)
value = np.zeros((jcp.max_cars()+1, jcp.max_cars()+1))
policy = value.copy().astype(int)
def apply_action(state, action):
    return [max(min(state[0] - action, jcp.max_cars()),0) , max(min(state[1] + action, jcp.max_cars()),0)]
def expected_reward(state, action):
    global value
    
    
    x = 0 #reward
    new_state = apply_action(state, action)    

    
    if action <= 0:
        x = x + jcp.moving_reward() * abs(action)
    else:
        x = x + jcp.moving_reward() * (action - 1)   
        
    
    if new_state[0] > 10:
        x = x + jcp.second_parking_lot_reward()
        
    if new_state[1] > 10:
        x = x + jcp.second_parking_lot_reward()

    for A in range(A.poisson.w, A.poisson.bet):
        for B in range(B.poisson.w, B.poisson.bet):
            for Ab in range(A.poisson.w, A.poisson.bet):
                for Bb in range(B.poisson.w, B.poisson.bet):
                    
                    # all four variables are independent of each other
                    ζ = A.poisson.vals[Aa] * B.poisson.vals[Ba] * A.poissonβ.vals[Ab] * B.poissonβ.vals[Bb]
                    
                    valid_requests_A = min(new_state[0], Aa)
                    valid_requests_B = min(new_state[1], Ba)
                    
                    rew = (valid_requests_A + valid_requests_B)*(jcp.credit_reward())
                    
                    #calculating the new state based on the values of the four random variables
                    new_s = [0,0]
                    new_s[0] = max(min(new_state[0] - valid_requests_A + Ab, jcp.max_cars()),0)
                    new_s[1] = max(min(new_state[1] - valid_requests_B + Bb, jcp.max_cars()),0)
                    
                    #Bellman's equation
                    x += j * (rew + jcp.y() * value[new_s[0]][new_s[1]])
                    
    return x
def policy_evaluation():
    
    global value
    
    e = policy_evaluation.e
    
    policy_evaluation.e /= 10 
    
    while(1):
        δ = 0
        
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                # value[i][j] denotes the value of the state [i,j]
                
                old_val = value[i][j]
                value[i][j] = expected_reward([i,j], policy[i][j])
                
                δ = max(δ, abs(value[i][j] - old_val))
                print('.', end = '')
                sys.stdout.flush()
        print(δ)
        sys.stdout.flush()
    
        if δ < e:
            break

policy_evaluation.e = 50
def policy_improvement():
    
    global policy
    
    policy_stable = True
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            old_action = policy[i][j]
            
            max_act_val = None
            max_act = None
            
            τ12 = min(i,5)       
            τ21 = -min(j,5)      
            
            for act in range(τ21,τ12+1):
                σ = expected_reward([i,j], act)
                if max_act_val == None:
                    max_act_val = σ
                    max_act = act
                elif max_act_val < σ:
                    max_act_val = σ
                    max_act = act
                
            policy[i][j] = max_act
            
            if old_action!= policy[i][j]:
                policy_stable = False
    
    return policy_stable
def save_policy():
    save_policy.counter += 1
    ax = sns.heatmap(policy, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('policy'+str(save_policy.counter)+'.svg')
    plt.close()
    
def save_value():
    save_value.counter += 1
    ax = sns.heatmap(value, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('value'+ str(save_value.counter)+'.svg')
    plt.close()
save_policy.counter = 0
save_value.counter = 0
while(1):
    policy_evaluation()
    ρ = policy_improvement()
    save_value()
    save_policy()
    if ρ == True:
        break