import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

#### Parameters #############
parser = argparse.ArgumentParser(description='Run the PD games in complex networks')
parser.add_argument('--network', type=str, default='ER', help='network types: Erdos-Renyi (ER) and Barabasi-Albert (BA) networks')
parser.add_argument('--N', type=int, default=1000, help='number of nodes in model networks')
parser.add_argument('--kave', type=float, default=8.0, help='average degree in model networks')
parser.add_argument('--game', type=str, default='PD', help='game type: Prisoner\'s dilemma (PD)')
parser.add_argument('--noise_K', type=float, default=0.1, help='intensity of selection for Fermi function')
parser.add_argument('--tmax', type=int, default=1000, help='maximum number of iterations in the PD games')
parser.add_argument('--nb_iter', type=int, default=100, help='number of iterations')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()

data_dir = "results"

# Normalize eps_set for colormap
norm = plt.Normalize(vmin=min(eps_set), vmax=max(eps_set))
cmap = plt.get_cmap('coolwarm')

eps_set = [0.2, 0.1, 0.0, -0.1, -0.2]
for eps in eps_set:
    if args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
        pattern = r'results\_{}\_payoff[\d\.]+\_{}\_tmax{}\_nbiter{}\_eps{}\_K{}_seed{}.csv'.format(
            args.game, args.network, args.tmax, args.nb_iter, eps, args.noise_K, args.seed)
    else:
        pattern = r'results\_{}\_payoff[\d\.]+\_{}\_N{}\_kave{}\_tmax{}\_nbiter{}\_eps{}\_K{}_seed{}.csv'.format(
            args.game, args.network, args.N, args.kave, args.tmax, args.nb_iter, eps, args.noise_K, args.seed)

    filenames = []
    for filename in os.listdir(data_dir):
        if re.match(pattern, filename):
            filenames.append(filename)

    b_values = sorted([float(re.search(r'payoff([\d.]+)_', filename).group(1)) for filename in filenames])
    b_values = [b for b in b_values if (b > 1.0) and (b < 2.0)]

    coop_ratios = []
    for payoff_b in b_values:
        if args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
            file_name = f'results/results_{args.game}_payoff{payoff_b}_{args.network}_tmax{args.tmax}_nbiter{args.nb_iter}_eps{eps}_K{args.noise_K}_seed{args.seed}.csv'
        else:
            file_name = data_dir + "/" + f'results_{args.game}_payoff{payoff_b}_{args.network}_N{args.N}_kave{args.kave}_tmax{args.tmax}_nbiter{args.nb_iter}_eps{eps}_K{args.noise_K}_seed{args.seed}.csv'

        df = pd.read_csv(file_name)
        tmp_coopratio = df.mean(axis=1).tolist()[-int(args.tmax * 0.1):]
        coop_ratios.append(sum(tmp_coopratio) / len(tmp_coopratio))

    color = cmap(norm(eps))
    plt.plot(b_values, coop_ratios, marker='o', markersize=6, linewidth=1, linestyle='--', label=f'${eps}$', color=color, markeredgecolor='#333333', markeredgewidth=0.5)

plt.xlabel('b')
plt.ylabel(r'$\rho$')
plt.ylim(-0.05,1.05)
plt.xlim(0.95,2.05)
plt.title(f'{args.network}')
plt.legend().set_title(f'$\epsilon$', prop={"size": 10})
plt.grid(True)

if args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
    plt.savefig(f'figure_coopratio_payoffb_wrt_eps_{args.game}_{args.network}_tmax{args.tmax}_K{args.noise_K}.png')
else:
    plt.savefig(f'figure_coopratio_payoffb_wrt_eps_{args.game}_{args.network}_N{args.N}_kave{args.kave}_tmax{args.tmax}_K{args.noise_K}.png')
