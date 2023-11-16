import argparse
import sys


def load_config():
    """
    Load configuration from command line
    """
    parser = argparse.ArgumentParser(
        description='Causal Structure Learning with Iterative LLM queries')
    parser.add_argument('--dataset', type=str, default="asia",
                        help="dataset name, can be one of ['asia', 'alarm', 'child', 'insurance', 'barley', 'mildew', 'cancer', 'water']")
    parser.add_argument('--model', type=str, default="gpt-4",
                        help="LLM model, can be one of ['gpt-3.5-turbo', 'gpt-4', 'sim-gpt']")
    parser.add_argument('--alg', type=str, default="HC",
                        help="CSL algorithm, can be one of ['CaMML', 'HC', 'softHC', 'hardMINOBSx', 'softMINOBSx']")
    parser.add_argument('--score', type=str, default="bdeu",
                        help="CSL algorithm, can be one of ['mml', 'bic', 'bdeu']")
    parser.add_argument('--data_index', type=int, default=1,
                        help="index of data, can be one of [1, 2, 3, 4, 5, 6]")
    parser.add_argument('--log_filepath', type=str, default=None,
                        help="log file path, print to screen as default")
    parser.add_argument('--datasize_index', type=int, default=0,
                        help="index of datasize params, can be one of [0, 1]")
    args = parser.parse_args()
    if args.log_filepath is not None:
        sys.stdout = open('out.txt', 'w')
    return args


if __name__ == '__main__':
    args = load_config()
    print(args)
