import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Parameters #############
parser = argparse.ArgumentParser(description='Run the PD games in complex networks')
parser.add_argument('--network', type=str, default='ER', help='network types: Erdos-Renyi (ER) and Barabasi-Albert (BA) networks')
parser.add_argument('--N', type=int, default=1000, help='number of nodes in model networks')
parser.add_argument('--kave', type=float, default=8.0, help='average degree in model networks')
parser.add_argument('--game', type=str, default='PD', help='game type: Prisoner\'s dilemma (PD)')
parser.add_argument('--payoff_b', type=float, default=1.4, help='Payoff parameter, advantage of defectors over cooperators for PD')
parser.add_argument('--noise_K', type=float, default=0.1, help='intensity of selection for Fermi function')
parser.add_argument('--tmax', type=int, default=1000, help='maximum number of iterations in the PD games')
parser.add_argument('--nb_iter', type=int, default=100, help='number of iterations')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--target', type=str, default='cooperate', help='target state')
args = parser.parse_args()


data_dir = "results"

suffix_set = ["", "_adaptive", "_random"]
label_dict = {
    "": "Adversarial",
    "_random": "Random",
    "_adaptive": "Li et al.",
}
symbol_set = ["o", "+", "x"]

for suffix, symbol in zip(suffix_set, symbol_set):
    pattern = r'results\_{}\_payoff{}+\_{}\_N{}\_kave{}\_tmax{}\_nbiter{}\_eps-?[\d\.]+\_K{}_seed{}{}.csv'.format(
    args.game, args.payoff_b, args.network, args.N, args.kave, args.tmax, args.nb_iter, args.noise_K, args.seed, suffix)

    filenames = []
    for filename in os.listdir(data_dir):
        if re.match(pattern, filename):
            filenames.append(filename)

    eps_values = sorted([float(re.search(r'eps(-?[\d.]+)_', filename).group(1)) for filename in filenames])
    
    if args.target == "cooperate":
        eps_values = [eps for eps in eps_values if eps >= 0]
    else:
        eps_values = [eps for eps in eps_values if eps <= 0]

    if suffix != "":
        if args.target == "cooperate":
            eps_values = np.array([0.0] + eps_values)
        else:
            eps_values = np.array(eps_values + [0.0])

    coop_ratios = []
    for eps in eps_values:
        if eps == 0.0:
            file_name = data_dir + "/" + f'results_{args.game}_payoff{args.payoff_b}_{args.network}_N{args.N}_kave{args.kave}_tmax{args.tmax}_nbiter{args.nb_iter}_eps{eps}_K{args.noise_K}_seed{args.seed}.csv'
        else:
            file_name = data_dir + "/" + f'results_{args.game}_payoff{args.payoff_b}_{args.network}_N{args.N}_kave{args.kave}_tmax{args.tmax}_nbiter{args.nb_iter}_eps{eps}_K{args.noise_K}_seed{args.seed}{suffix}.csv'
        df = pd.read_csv(file_name)
        tmp_coopratio = df.mean(axis=1).tolist()[-int(args.tmax * 0.1):]
        coop_ratios.append(sum(tmp_coopratio) / len(tmp_coopratio))

    if args.target != "cooperate":
        eps_values = np.abs(eps_values)[::-1]
        coop_ratios = coop_ratios[::-1]

    if args.target == "defect" and suffix == "_adaptive":
        pass
    else:
        plt.plot(eps_values, coop_ratios, color="black", marker=symbol, markersize=6, linewidth=1, linestyle='--', label=f'{label_dict[suffix]}')

plt.xlabel(r'Perturbation strength $|\epsilon|$')
plt.ylabel(r'Proportion of cooperators $\rho$')
plt.ylim(-0.05,1.05)
plt.title(f'{args.network}')
plt.legend()
plt.grid(True)
plt.savefig(f'figure_coopratio_eps_{args.game}_payoff{args.payoff_b}_{args.target}_{args.network}_N{args.N}_kave{args.kave}_tmax{args.tmax}.png')
