import treelib
import numpy as np
from ttt_env import TicTacToe
import math

"""
usefull resources: 
 http://matthewdeakos.me/2018/03/10/monte-carlo-tree-search/
 https://www.youtube.com/watch?v=UXW2yZndl7U
"""
C = 2
STEPS = 10000

env = TicTacToe()

game_tree = treelib.Tree()
game_tree.create_node("0", "0", data = {"visited": 0, "total": 0, "state": np.zeros((3,3)),
                                        "player": 1, "terminal": False, "action": None})

def get_active(active_player):
    return 1 if active_player == 2 else 1


def selection(cur_node_id):
    children = game_tree.children(cur_node_id)
    cur_data = game_tree.get_node(cur_node_id).data

    avg_v = cur_data["total"]
    l_vis = np.log(cur_data["visited"])

    best = None
    for c in children:
        # avoid 0 div
        ucb1 = avg_v + C * math.sqrt(l_vis / (game_tree.get_node(c).data["visited"] + 0.0000001)) 
        if best is None or ucb1 > best[0]:
            best = (ucb1, c)

    return best[1]


def light_rollout(cur_node):
    env.state = cur_node.data["state"]
    player = cur_node.data["player"]
    while True:
        if env.get_done()[1]: # tie
            return 0, player
        elif env.get_done()[0]:
            return 1, player

        env.update_board(np.random.choice(env.get_actions()), player)
        player = get_active(player)


def recursive_update(node, r, p):
    node.data["visited"] += 1
    node.data["total"] += r if node.data["player"] == p else 0
    if not node.is_root():
        recursive_update(game_tree.get_node(node.bpointer), r, p)

def train():

    id_counter = 1
    for _ in range(STEPS):
        current = game_tree.get_node("0")
        while not leaf:
            if current.data["terminal"]:
                break
            leaf = len(game_tree.children(current.identifier)) == 0
            if not leaf:
                current = selection(current.identifier) 

        if current.data["visited"] == 0:
            r, p = light_rollout(current)
            recursive_update(current, r, p)
        else:
            if current.data["terminal"]:
                current.data["visited"] += 1
                env.field = current.data["state"]
                val = 1 if env.get_done()[1] else 0
                p = current.data["player"]
                recursive_update(current, val, p)

            env.field = current.data["state"]
            acts = env.get_actions()
            for act in acts:
                player = get_active(current.data["player"])
                obs = env.update_board(act, player=player)
                terminal, _ = env.get_done()
                name = f"{id_counter}_{act}"
                game_tree.create_node(name, name, current.identifier, data= {"visited": 0, "total": 0,
                    "state": obs, "player": player, "terminal": terminal, "action": act})
                id_counter += 1

            childs = game_tree.children(current.identifier)
            c = childs[0]
            r, p = light_rollout(c)
            recursive_update(c, r, p)






    