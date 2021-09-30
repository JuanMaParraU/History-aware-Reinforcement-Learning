import re
import csv
import numpy as np
import sys
from pkg_resources._vendor.pyparsing import line
print(sys.version)
import matplotlib.pyplot as plt
RE_Episode=re.compile(r"Episode: (\d+)")
RE_Reward = re.compile(r'Reward by drone: (\d+)')
RE_RewardTotal = re.compile(r'Reward total: (\d+)')
RE_Step = re.compile(r'Step: (\d+)')
RE_Drone = re.compile(r'Drone: (\d+)')
# The lists where we will store results.
epi = []#episode
reward = [] #Actual reward by drone 
rewardTotal = [] #Actual total
step = [] #step or iteration
drone = [] #Drone number
epi0 = [] #Episode for drone 0
epi1 = [] #Episode for drone 1 
reward0 = [] #Reward for drone 0
rewardTotal0 = [] #Reward for drone 0
reward1 = [] #Reward for drone 1
rewardTotal1 = [] #Reward for drone 0
def ImprovementsAnalyzer(nfile):
 try:                            
    with open(nfile, 'rt') as in_file:        # open file for reading text.
      for line in in_file:
          e=RE_Episode.search(line)
          r=RE_Reward.search(line)
          rt=RE_RewardTotal.search(line)
          s=RE_Step.search(line)
          d=RE_Drone.search(line)
          if e is not None:# If substring search finds a match,
            epi.append(e.group(1)) 
          if r is not None:
            reward.append(r.group(1))
          if s is not None:
            step.append(s.group(1))
          if d is not None:# If substring search finds a match,
            drone.append(d.group(1))   
          if rt is not None:# If substring search finds a match,
            rewardTotal.append(rt.group(1))             
    print(len(epi),len(reward),len(step),len(drone),len(rewardTotal))
 except IOError:                   # If log file not found,
  print("Log file or search not found.") 
def CsvCreator(csvname):
 try: 
  with open(csvname,'w') as output:
   writer = csv.writer(output,lineterminator='\n')	
   c = np.c_[epi,step,drone,reward,rewardTotal]
   writer.writerow(["Episode", "Step","Drone", "Reward drone","Reward total"])
   writer.writerows(c)
   print (c)
 except IOError:                   # If log file not found,
  print("CSV file not found.")
def graphReward():
    index=[i for i, a in enumerate(drone)if a=="0"]
    for ind in index:
        reward0.append(float(reward[ind]))
        epi0.append(float(epi[ind]))
        rewardTotal0.append(float(rewardTotal[ind]))
    r0 = np.asarray(reward0)
    rt0= np.asarray(rewardTotal0)
    p0 = np.asarray(epi0)
    plt.figure(1)
    plt.title('Rewards by episode drone 0')
    plt.xlabel('Episode')
    plt.ylabel('Users connected')
    plt.plot(p0,r0,'r',label = 'Drone reward')
    plt.plot(p0,rt0,'b',label = 'Total reward')
    plt.legend()
    index=[i for i, a in enumerate(drone)if a=="1"]
    for ind in index:
        reward1.append(float(reward[ind]))
        epi1.append(float(epi[ind]))
        rewardTotal1.append(float(rewardTotal[ind]))
    r1 = np.asarray(reward1)
    rt1 = np.asarray(rewardTotal1)
    p1 = np.asarray(epi1)
    plt.figure(2)
    plt.title('Rewards by episode drone 1')
    plt.xlabel('Episode')
    plt.ylabel('Users connected')
    plt.plot(p1,r1,'r',label = 'Drone reward')
    plt.plot(p1,rt1,'b',label = 'Total reward')
    plt.legend()
    plt.show()        
    
if __name__ == '__main__':
 ImprovementsAnalyzer('/home/CAMPUS/parra-uj/Documents/dqn-output.txt')
 CsvCreator('./CSVlog1.csv')             
 graphReward()
