#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import halite helpers for the game, and the rest for the profiler
from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from collections import defaultdict
from heapq import nlargest
from matplotlib import cm


# In[2]:


# Game Profiler Claass
class GameProfiler:
    """
    Logs higher level statistics for the game to be examined 
    after the game is finished.

    Items tracked for self and opponents:
        - Total halite at each turn
        - Ship Actions and Halite
        - Shipyard Actions
    """

    def __init__(self):
        self.gamelog = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        # up to five level of keys dict[one][two][three][four][five]

    def take_snapshot(self,BOARD,print_board=False):
        if print_board:
            print('Step:',BOARD.observation['step'])
            print(BOARD.current_player.next_actions)        
            print(BOARD)

        for player_id,player in BOARD.players.items():
            player_id = "p" + str(player_id) 
            
            self.gamelog[player_id]['player'][BOARD.observation['step']]['total_halite'] = player.halite
            for ship in player.ships:
                self.gamelog[player_id]['ship'][ship.id][BOARD.observation['step']]['action'] = ship.next_action
                self.gamelog[player_id]['ship'][ship.id][BOARD.observation['step']]['halite'] = ship.halite
            for yard in BOARD.current_player.shipyards:
                self.gamelog[player_id]['yard'][yard.id][BOARD.observation['step']] ['action'] = yard.next_action


# In[3]:


def profiler_to_panda(shiplog):

    halite_stats = pd.DataFrame(index=range(0,len(shiplog['p0']['player'].keys())))
    game_steps = list(halite_stats.index)

    for player, player_vars in shiplog.items():
        #print('name:',player)
        halite = player_vars['player']
        ships = player_vars['ship']
        yards = player_vars['yard']
        # obtain halite over steps
        hal = list()
        for step,h in halite.items():
            hal.append(h['total_halite'])
            #print('step: {} halite: {}'.format(step,h['total_halite']))
        halite_stats[player] = hal
        # obtain ship halite per steps
        for shipname, shipstats in ships.items():
            s = list()
            #print('name: {}'.format(shipname))
            for stp in game_steps:
                if stp in shipstats:
                    s.append(shipstats[stp]['halite'])
                else:
                    s.append(None)
            halite_stats[player+"-"+shipname] = s

    return halite_stats, list(PROFILER.gamelog.keys())


# In[4]:


# plotting game profile

player_maps =['cool','summer','autumn','winter']
def plot_game(halite_stats,players,agent_names=[]):

    num_subplots = len(players) + 1
    fig, ax = plt.subplots(num_subplots,figsize=(12,2*num_subplots), sharex=True)

    for col in range(0,len(players)):
        cmap = plt.get_cmap(player_maps[col])
        ax[0].plot(halite_stats.loc[:, players[col]],color=cmap(0),label=players[col])
    ax[0].set_ylabel('Total Halite')
    ax[0].legend(loc='upper left',title='Agents')
    ax[0].set_title('Game Profiler')


    ship_names = list(halite_stats.columns.difference(players))
    for x in range(0,len(players)):

        cols = [name for name in ship_names if name.startswith(players[x])]
        cmap = plt.get_cmap(player_maps[x])
        colors = [cmap(i) for i in np.linspace(0, .7, len(cols))]
        for col in range(0,len(cols)):
            ax[x+1].plot(halite_stats.loc[:, cols[col]],color=colors[col])
        ymax = halite_stats.loc[:, cols].max().max()
        for col in range(0,len(cols)):
            startship = halite_stats[halite_stats[cols[col]] == np.min(halite_stats.loc[:, cols[col]])].index.values[0]
            ax[x+1].vlines(startship, 0, ymax, colors=colors[col], linestyles='dashed',alpha=.2)
        agent_name = ""
        if len(agent_names) == len(players):
            agent_name = ": " + agent_names[x]
        ax[x+1].text(0.01, 0.92, '{}{}\nShips: {}'            .format(players[x],agent_name,len(cols)), horizontalalignment='left', 
        verticalalignment='top', transform=ax[x+1].transAxes)
        ax[x+1].set_ylabel('Ship Halite')
        if x == len(players)-1:
            ax[x+1].set_xlabel('Game Steps')


# In[5]:


# Returns best direction to move from one position (fromPos) to another (toPos)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
def getDirTo(fromPos, toPos, size):
    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)
    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)
    if fromY < toY: return ShipAction.NORTH
    if fromY > toY: return ShipAction.SOUTH
    if fromX < toX: return ShipAction.EAST
    if fromX > toX: return ShipAction.WEST

# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
STARTED_SHIP_STATES = {}

# Returns the commands we send to our ships and shipyards
def getting_started_agent(obs, conf):
    size = conf.size
    board = Board(obs, conf)
    me = board.current_player

    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT
    
    for ship in me.ships:
        if ship.next_action == None:
            
            ### Part 1: Set the ship's state 
            if ship.halite < 200: # If cargo is too low, collect halite
                STARTED_SHIP_STATES[ship.id] = "COLLECT"
            if ship.halite > 500: # If cargo gets very big, deposit halite
                STARTED_SHIP_STATES[ship.id] = "DEPOSIT"
                
            ### Part 2: Use the ship's state to select an action
            if STARTED_SHIP_STATES[ship.id] == "COLLECT":
                # If halite at current location running low, 
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    neighbors = [ship.cell.north.halite, ship.cell.east.halite, 
                                 ship.cell.south.halite, ship.cell.west.halite]
                    best = max(range(len(neighbors)), key=neighbors.__getitem__)
                    ship.next_action = directions[best]
            if STARTED_SHIP_STATES[ship.id] == "DEPOSIT":
                # Move towards shipyard to deposit cargo
                direction = getDirTo(ship.position, me.shipyards[0].position, size)
                if direction: ship.next_action = direction
                
    PROFILER.take_snapshot(board)            
    return me.next_actions


# In[6]:


PROFILER = GameProfiler()
env = make("halite", debug=True,configuration={'episodeSteps': 200,'width':10})
agent_count = 2
env.reset(agent_count);
env.run([getting_started_agent,"random"])
halite_stats, players = profiler_to_panda(PROFILER.gamelog)
names = ["getting_started_agent", "random"]
plot_game(halite_stats,players,agent_names=names)
#env.render(mode="ipython", width=400, height=300)


# In[7]:


PROFILER = GameProfiler()
env = make("halite", debug=True)
env.reset();
env.run([getting_started_agent,"random","random","random"])
halite_stats, players = profiler_to_panda(PROFILER.gamelog)
names = ["getting_started_agent", "random 1","random 2","random 3"]
plot_game(halite_stats,players,agent_names=names)


# In[8]:


## GLOBAL VARIABLES

action_for_direction = {
    (1,0): ShipAction.EAST,
    (-1,0):ShipAction.WEST,
    (0,1): ShipAction.NORTH,
    (0,-1):ShipAction.SOUTH,
    (0,0): None 
}

# Directions a ship can move
POINT_DIRECTIONS = [(0,1),(1,0),(0,-1),(-1,0)]
ALL_DIRECTIONS = [(1,0),(-1,0),(0,1),(0,-1),(0,0)]
OPPOSITE_DIRECTION = {(1,0):(-1,0), (-1,0):(1,0), (0,1):(0,-1), (0,-1):(0,1)}

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
MY_SHIP_STATES = defaultdict(dict)
FULL_SHIP = 600
MAX_SHIPS = 3
GOALS = set()

def add_and_normalize(a, b, size):
    ax, ay = a
    bx, by = b
    return ((ax + bx) % size, (ay + by)% size)

def manhattan_on_donut(start, end, size ):
    dx = np.min([ abs(start[0] - end[0]), size - abs(end[0] - start[0]) ])
    dy = np.min([ abs(start[1] - end[1] ), size - abs(end[1] - start[1]) ])

    dir_x = np.argmin([abs( start[0] - end[0]), size - abs( end[0] - start[0] ) ])
    dir_y = np.argmin([ abs( start[1] - end[1] ), size - abs( end[1] - start[1] ) ])
    return dx, dy, dir_x, dir_y


def get_halite_goal(num_ships,BOARD):
    # find the top num_ship cells with most halite
    most_halite = nlargest(num_ships+1, BOARD.cells.values(), key=lambda x: x.halite)
    # return the position of the first one that isn't already a goal
    for cell in most_halite:
        if not cell.position in GOALS:
            return cell.position
        
def direction_between(start, end, size):

    def check_for_opposite(check,direction):
        if check == 1:
            return OPPOSITE_DIRECTION[direction]
        else:
            return direction
    
    if start == end:
        return (0,0) # same point, stay there

    dx, dy, dir_x, dir_y = manhattan_on_donut(start,end,size)  
    if dx > dy:
        # move in x direction
        if start[0] - end[0] < 0:  
            return check_for_opposite(dir_x,(1,0)) # maybe move right
        else:
            return check_for_opposite(dir_x,(-1,0)) # maybe move left
    else:
        # move in y direction
        if start[1] - end[1] < 0:
            return check_for_opposite(dir_y,(0,1)) # maybe move up
        else:
            return check_for_opposite(dir_y,(0,-1)) # maybe move down
    

######## AGENT STARTS HERE ################################
def mine_most_halite(obs, conf):
    size = conf.size
    BOARD = Board(obs, conf)
    me = BOARD.current_player

    # positions that my ships shouldn't go (i.e., other shipyard, my ship there)
    RESERVED_POSITIONS = set()
    for opponent in BOARD.opponents:
        for yard in opponent.shipyards:
            RESERVED_POSITIONS.add(yard.position)

    # remove dead ship goals
    my_ship_ids = set(MY_SHIP_STATES.keys())
    live_ship_ids = set(me.ship_ids)
    dead_ids = my_ship_ids - live_ship_ids
    for id in dead_ids:
        shipgoal = MY_SHIP_STATES[id]['GOAL']
        GOALS.discard(shipgoal)
        MY_SHIP_STATES.pop(id)

    # SHIP AND SHIPYARD SPAWNING
    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN
    elif len(me.shipyards) > 0 and len(me.ships) < MAX_SHIPS:
        if me.shipyards[0].cell.ship_id == None:
            me.shipyards[0].next_action = ShipyardAction.SPAWN
    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT
    
    for ship in me.ships:
        if ship.next_action == None:

            ## Step 0: if the ship is new, add action and goal
            if not ship.id in MY_SHIP_STATES.keys():
                MY_SHIP_STATES[ship.id]['ACTION'] = 'FIND'
                MY_SHIP_STATES[ship.id]['GOAL'] = get_halite_goal(len(me.ship_ids),BOARD)

            ### Step 1: Potentially update goals
            at_goal = ship.position == MY_SHIP_STATES[ship.id]['GOAL']
            current_action = MY_SHIP_STATES[ship.id]['ACTION']
            
            if at_goal:
                if current_action == 'FIND':
                    MY_SHIP_STATES[ship.id]['ACTION'] = 'COLLECT'
                    GOALS.discard(MY_SHIP_STATES[ship.id]['GOAL'])
                    MY_SHIP_STATES[ship.id]['GOAL'] = (-1,-1)

                elif current_action == 'DEPOSIT':
                    MY_SHIP_STATES[ship.id]['ACTION'] = 'FIND'
                    MY_SHIP_STATES[ship.id]['GOAL'] = get_halite_goal(len(me.ship_ids),BOARD)
                    GOALS.add(MY_SHIP_STATES[ship.id]['GOAL'])
            else:
                if ship.halite > FULL_SHIP: # If cargo gets very big, deposit halite
                    MY_SHIP_STATES[ship.id]['ACTION'] = "DEPOSIT"
                    # TODO: account for more than one shipyard!
                    if me.shipyards[0]:
                        MY_SHIP_STATES[ship.id]['GOAL'] = me.shipyards[0].position

                
            ### Step 2: Use the ship's state to select an action
            next_move = (0,0)
            # if FIND or DEPOSIT, move towards GOAL
            if MY_SHIP_STATES[ship.id]['ACTION'] == 'FIND' or                         MY_SHIP_STATES[ship.id]['ACTION'] == 'DEPOSIT': 
                next_move = direction_between(ship.position,MY_SHIP_STATES[ship.id]['GOAL'],size) 
            # if COLLECT, mine or find local maximal halite
            if MY_SHIP_STATES[ship.id]['ACTION'] == "COLLECT":
                # If halite at current location running low, 
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    neighbors = [ship.cell.north.halite, ship.cell.east.halite, 
                                 ship.cell.south.halite, ship.cell.west.halite]
                    best = max(range(len(neighbors)), key=neighbors.__getitem__)
                    next_move = POINT_DIRECTIONS[best]    

            ## Part 3: avoid collisions
            next_position = add_and_normalize(ship.position,next_move,size)
            if next_position in RESERVED_POSITIONS:
                keyset = set(action_for_direction.keys())
                keyset.remove(next_move)
                keylist = list(keyset)
                next_move = keylist[np.random.choice(len(keylist))]

            RESERVED_POSITIONS.add(next_position)
            ship.next_action = action_for_direction[next_move]

    return me.next_actions


# In[9]:


PROFILER = GameProfiler()
env = make("halite", debug=True)
env.reset();
env.run([getting_started_agent,mine_most_halite,"random","random"])
halite_stats, players = profiler_to_panda(PROFILER.gamelog)
names = ["getting_started_agent", "mine_most_halite","random 1","random 2"]
plot_game(halite_stats,players,agent_names=names)


# In[10]:


MEAN_SHIP_STATES = defaultdict(dict)
FULL_MEAN_SHIP = 300
MAX_MEAN_SHIPS = 4
MEAN_GOALS = set()

def get_ship_locations(BOARD):
    player_halite = list()
    o_ship = list()
    halite = list()
    position = list()
    for opponent in BOARD.opponents:
        for ship in opponent.ships:
            player_halite.append(opponent.halite)
            o_ship.append(ship.id)
            halite.append(ship.halite)
            position.append(ship.position)

    other_ships = pd.DataFrame(data={'player_halite':player_halite,'ship':o_ship,'halite':halite,'position':position})
    return other_ships

def get_ship_distance(other_ships, ship,size):

    def get_distance(o_pos,s_pos):
        dx, dy, _, _ = manhattan_on_donut(s_pos,o_pos,size)
        return dx + dy

    if len(other_ships) == 0:
        return other_ships
    else:
        dists = list()
        for other_pos in other_ships['position']:
            dists.append(get_distance(other_pos,ship.position))
        other_ships[ship.id] = dists
        return other_ships

def get_goal_ship(other_ships,ship):
    if len(other_ships) == 0:
        return None, None
    else:
        #other_ships.sort_values(['player_halite',ship.id, 'halite'], ascending=[False,True, False],inplace=True)
        other_ships.sort_values([ship.id, 'halite'], ascending=[True, False],inplace=True)
        for x in range(0,len(other_ships)):
            if other_ships.loc[x]['halite'] > ship.halite:
                if not other_ships.loc[x]['ship'] in MEAN_GOALS:
                    MEAN_GOALS.add(other_ships.loc[x]['ship'])
                    return other_ships.loc[x]['position'], other_ships.loc[x]['ship']
        return None, None

def agressive_agent(obs, conf):
    size = conf.size
    BOARD = Board(obs, conf)
    me = BOARD.current_player

    # positions that my ships shouldn't go (i.e., other shipyard, my ship there)
    RESERVED_POSITIONS = set()
    for opponent in BOARD.opponents:
        for yard in opponent.shipyards:
            RESERVED_POSITIONS.add(yard.position)

    # get other ships and locations
    other_ships = get_ship_locations(BOARD)

    # remove dead ship goals
    # if my ship died
    my_ship_ids = set(MEAN_SHIP_STATES.keys())
    live_ship_ids = set(me.ship_ids)
    dead_ids = my_ship_ids - live_ship_ids
    for id in dead_ids:
        shipgoal = MEAN_SHIP_STATES[id]['GOAL_SHIP']
        MEAN_GOALS.discard(shipgoal)
        MEAN_SHIP_STATES.pop(id)
    # if other ship died
    for id in list(MEAN_GOALS):
        if id not in other_ships['ship']:
            MEAN_GOALS.discard(id)

    # SHIP AND SHIPYARD SPAWNING
    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN
    elif len(me.shipyards) > 0 and len(me.ships) < MAX_MEAN_SHIPS:
        if me.shipyards[0].cell.ship_id == None:
            me.shipyards[0].next_action = ShipyardAction.SPAWN
    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT
    
    for ship in me.ships:
        if ship.next_action == None:

            # Step -1: find distances
            other_ships = get_ship_distance(other_ships,ship,size)

            ## Step 0: if the ship is new, add action and goal
            if not ship.id in MEAN_SHIP_STATES.keys():
                MEAN_SHIP_STATES[ship.id]['ACTION'] = 'HUNT'
                goal_pos, goal_ship = get_goal_ship(other_ships,ship)
                MEAN_SHIP_STATES[ship.id]['GOAL_POS'] = goal_pos
                MEAN_SHIP_STATES[ship.id]['GOAL_SHIP'] = goal_ship
                if not goal_ship == None:
                    MEAN_GOALS.add(goal_ship)

            ### Step 1: Potentially update goals
            at_goal = False
            if MEAN_SHIP_STATES[ship.id]['GOAL_POS']:
                at_goal = ship.position == MEAN_SHIP_STATES[ship.id]['GOAL_POS']

            if at_goal:
                # if caught ship or deposited, hunt
                MEAN_SHIP_STATES[ship.id]['ACTION'] = 'HUNT'
                goal_pos, goal_ship = get_goal_ship(other_ships,ship)
                MEAN_SHIP_STATES[ship.id]['GOAL_POS'] = goal_pos
                MEAN_SHIP_STATES[ship.id]['GOAL_SHIP'] = goal_ship
            else:
                if ship.halite > FULL_MEAN_SHIP: # If cargo gets very big, deposit halite
                    MEAN_SHIP_STATES[ship.id]['ACTION'] = "DEPOSIT"
                    # TODO: account for more than one shipyard!
                    MEAN_SHIP_STATES[ship.id]['GOAL_POS'] = me.shipyards[0].position
                    MEAN_SHIP_STATES[ship.id]['GOAL_SHIP'] = None
                elif not MEAN_SHIP_STATES[ship.id]['GOAL_POS'] or                             not MEAN_SHIP_STATES[ship.id]['GOAL_SHIP'] in MEAN_GOALS:
                    # if no previous goal, or is ship is gone, new goals
                    goal_pos, goal_ship = get_goal_ship(other_ships,ship)
                    MEAN_SHIP_STATES[ship.id]['GOAL_POS'] = goal_pos
                    MEAN_SHIP_STATES[ship.id]['GOAL_SHIP'] = goal_ship
                else:
                    # update location of goal
                    goal_ship = MEAN_SHIP_STATES[ship.id]['GOAL_SHIP']
                    newloc = other_ships['position'][other_ships['ship'] == goal_ship].values[0]
                    MEAN_SHIP_STATES[ship.id]['GOAL_POS'] = newloc
                    #print('updating position')

                
            ### Step 2: Use the ship's state to select an action
            next_move = (0,0)
            # if DEPOSIT, move towards GOAL
            if MEAN_SHIP_STATES[ship.id]['ACTION'] == 'DEPOSIT':  
                next_move = direction_between(ship.position,MEAN_SHIP_STATES[ship.id]['GOAL_POS'],size)  
            # if COLLECT, mine or find local maximal halite
            if MEAN_SHIP_STATES[ship.id]['ACTION'] == "HUNT":
                # if no one to hunt, wander
                if not MEAN_SHIP_STATES[ship.id]['GOAL_POS']:
                    idx = np.random.choice(len(POINT_DIRECTIONS))
                    next_move = POINT_DIRECTIONS[idx]
                else:
                    next_move = direction_between(ship.position,MEAN_SHIP_STATES[ship.id]['GOAL_POS'],size)  
                
            ## Part 3: avoid collisions
            neighbor = ship.position
            next_position = add_and_normalize(ship.position,next_move,size)
            if next_position in RESERVED_POSITIONS:
                neighbors = set([ship.cell.north.position, ship.cell.east.position, 
                            ship.cell.south.position, ship.cell.west.position])
                neighbors = neighbors - RESERVED_POSITIONS
                neighbors.discard(me.shipyards[0].position)
                if len(neighbors) > 0:
                    neighbors = list(neighbors)
                    choice = np.random.choice(len(neighbors))
                    neighbor = neighbors[choice]
                next_move = direction_between(ship.position,neighbor,size)
                next_position = add_and_normalize(ship.position,next_move,size)

            RESERVED_POSITIONS.add(next_position)
            ship.next_action = action_for_direction[next_move]

    return me.next_actions


# In[11]:


PROFILER = GameProfiler()
env = make("halite", debug=True)
env.reset();
env.run([getting_started_agent,mine_most_halite,agressive_agent,"random"])
halite_stats, players = profiler_to_panda(PROFILER.gamelog)
names = ["getting_started_agent", "mine_most_halite","agressive_agent","random"]
plot_game(halite_stats,players,agent_names=names)


# In[12]:



env.render(mode="ipython", width=400, height=300)


# In[ ]:




