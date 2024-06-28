import argparse
import numpy as np
import networkx as nx
import pandas as pd
import random
import os
from tqdm import tqdm
from multiprocessing import Pool

from utils import load_network_data

#### Parameters #############
parser = argparse.ArgumentParser(description='run evolutionary game in complex networks')
parser.add_argument('--network', type=str, default='ER', help='network types: Erdos-Renyi (ER), Barabasi-Albert (BA), and  Watts-Strogatz (WS) networks')
parser.add_argument('--N', type=int, default=1000, help='number of nodes in model networks')
parser.add_argument('--kave', type=float, default=8.0, help='average degree in model networks')
parser.add_argument('--game', type=str, default='PD', help='game type: Prisoner\'s dilemma (PD)')
parser.add_argument('--payoff_b', type=float, default=1.4, help='payoff parameter: advantage of defectors over cooperators in PD')
parser.add_argument('--noise_K', type=float, default=0.1, help='noise intensity in strategy adoption')
parser.add_argument('--tmax', type=int, default=1000, help='maximum number of iterations in the game')
parser.add_argument('--eps', type=float, default=0.2, help='epsilon: perturbation strength for attacks')
parser.add_argument('--nb_iter', type=int, default=100, help='number of iterations')
parser.add_argument('--seed', type=int, default=123, help='random seed for reproducibility')
parser.add_argument('--random', action='store_true', help='perform random attacks with strength eps')
parser.add_argument('--adaptive', action='store_true', help='use adaptive adjustment of link weights (Li et al. Appl Math Comput. 361:810-820, 2019)')
args = parser.parse_args()

# set payoff matrix
payoff_params = {
    'PD': {'T': args.payoff_b, 'R': 1.0, 'P': 0.0, 'S': 0.0},
}

state_dict = {
    'cooperate': 1,
    'defect': -1,
}

def FermiRule(x):
    return 1.0 / (1.0 + np.exp(min(500.0, x / args.noise_K)))

def run_iteration(n):
    np.random.seed(args.seed + n)
    
    # generate a network
    if args.network == 'BA':
        # Barabasi-Albert model
        g = nx.barabasi_albert_graph(args.N, int(args.kave / 2), seed=args.seed + n)
    elif args.network  == 'ER':
        # Erdos-Renyi model
        g = nx.gnm_random_graph(args.N, int(args.kave * args.N / 2), directed=False, seed=args.seed + n)
    elif args.network == 'WS':
        # Watts-Strogatz model
        pws = 0.05
        g = nx.watts_strogatz_graph(args.N, int(args.kave), pws, seed=args.seed + n)
    elif args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
        # real-world social network
        g = load_network_data(args.network)
        args.N = g.number_of_nodes() # update
    else:
        raise ValueError("invalid network")

    # convert to directed network format
    g = nx.DiGraph(g)

    # Initialize actions
    coop_ratios = []
    actions = np.array(['cooperate'] * (args.N // 2) + ['defect'] * (args.N - args.N // 2))
    np.random.shuffle(actions)
    nx.set_node_attributes(g, dict(zip(g.nodes, actions)), 'action')
    for v in g.nodes:
        for w in g.neighbors(v):
            g.edges[w, v]['weight'] = 1.0

    # compute cooperation ratio at time 0
    coop_ratios.append(np.mean(actions == 'cooperate'))

    for t in range(1, args.tmax):
        next_actions = {}

        # compute total payoff for game at time 0
        for v in g.nodes:
            if g.nodes[v]['action'] == 'cooperate':
                g.nodes[v]['total_payoff'] = np.sum(np.array([payoff_params[args.game]['R'] * g.edges[w, v]['weight'] if g.nodes[w]['action'] == 'cooperate' else payoff_params[args.game]['S'] * g.edges[w, v]['weight'] for w in g.neighbors(v)]))
            else:
                g.nodes[v]['total_payoff'] = np.sum(np.array([payoff_params[args.game]['T'] * g.edges[w, v]['weight'] if g.nodes[w]['action'] == 'cooperate' else payoff_params[args.game]['P'] * g.edges[w, v]['weight'] for w in g.neighbors(v)]))

        in_weighted_degree = g.in_degree(weight='weight')
        for i in g.nodes:
            next_actions[i] = g.nodes[i]['action']
            neighbors = np.array(list(g.neighbors(i)))
            degree_i = len(neighbors)

            if degree_i > 0:
                if abs(args.eps) > 0.0:
                    # note: assuming that all link weights have a value of 1.0
                    if args.random:
                        # random perturbations
                        weights = np.random.choice([1.0-args.eps, 1.0+args.eps], size=degree_i)

                    elif args.adaptive:
                        # link weight adaptively adjusted by comparing the individual payoff with her/his surrounding environment
                        weights = np.array([g.edges[j, i]['weight'] for j in neighbors])
                        neighbors_payoffs = np.array([g.nodes[j]['total_payoff'] for j in neighbors])
                        weights = weights + 0.01 * np.sign(g.nodes[i]['total_payoff'] - np.mean(neighbors_payoffs))
                        weights = np.clip(weights, 1.0-abs(args.eps), 1.0+abs(args.eps))

                    elif args.simple:
                        # simple computation of adversarial perturbations
                        sign_gradients = np.ones(degree_i)
                        neighbor_states = np.array([state_dict[g.nodes[j]['action']] for j in neighbors])
                        sign_gradients[np.where(neighbor_states == -1)] = -1
                        weights = 1.0 + args.eps * sign_gradients
                    
                    else:
                        # compute adversarial perturbations
                        neighbors_payoffs = np.array([g.nodes[j]['total_payoff'] for j in neighbors])
                        probs = np.array([FermiRule(g.nodes[i]['total_payoff'] - p) for p in neighbors_payoffs])
                        diff_state = np.array([state_dict[g.nodes[i]['action']] - state_dict[g.nodes[j]['action']] for j in neighbors])
                        tmp = probs * diff_state
                        
                        if degree_i > 1:
                            gradients = tmp - (np.sum(tmp) - tmp) / (degree_i - 1)
                        else:
                            gradients = tmp
                        
                        weights = 1.0 - args.eps * np.sign(gradients)

                    for j, weight in zip(neighbors, weights):
                        g.edges[j, i]['weight'] = weight
                    
                    j = np.random.choice(neighbors, p=weights/weights.sum())

                else:
                    j = np.random.choice(neighbors)

                if FermiRule(g.nodes[i]['total_payoff'] - g.nodes[j]['total_payoff']) > np.random.random():
                    next_actions[i] = g.nodes[j]['action']

        # Update actions for the next step
        node_list = list(g.nodes)
        next_actions_array = np.array([next_actions[i] for i in node_list])
        nx.set_node_attributes(g, dict(zip(node_list, next_actions_array)), 'action')

        # compete cooperation ratio at time t
        coop_ratios.append(np.mean(next_actions_array == 'cooperate'))

    return coop_ratios


if __name__ == '__main__':
    # Initialize output data
    df = pd.DataFrame(index=range(args.tmax))
    # set output file name
    if args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
        file_name = f'results/results_{args.game}_payoff{args.payoff_b}_{args.network}_tmax{args.tmax}_nbiter{args.nb_iter}_eps{args.eps}_K{args.noise_K}_seed{args.seed}'
    else:
        file_name = f'results/results_{args.game}_payoff{args.payoff_b}_{args.network}_N{args.N}_kave{args.kave}_tmax{args.tmax}_nbiter{args.nb_iter}_eps{args.eps}_K{args.noise_K}_seed{args.seed}'
        
    if args.random:
        file_name += '_random'
    elif args.adaptive:
        file_name += '_adaptive'

    file_name += '.csv'
    
    np.random.seed(args.seed)

    if os.path.exists(file_name):
        print("already computed.")
    else:
        # evolutionary game dynamics
        with Pool() as pool:
            results = list(tqdm(pool.imap(run_iteration, range(args.nb_iter)), total=args.nb_iter, desc="Iteration"))

        for n, coop_ratios in enumerate(results):
            df["iter" + str(n)] = coop_ratios

        # save ouput data 
        df.to_csv(file_name, index=False)
