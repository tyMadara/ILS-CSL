# Causal Structure Learning Supervised by Large Language Model

Welcome to the repository of code and materials of the research on Causal Structure Learning Supervised by Large Language Model.

## Installation

Follow these steps to install and set up the necessary environments for running the project:

1. **Download repository**: Download all the necessary files to your local machine.

2. **Install Graphviz**: If not installed, run the following command:

   ```bash
   sudo apt-get install graphviz libgraphviz-dev gcc
   ```

3. **Navigate to the project folder**:

   ```bash
   cd ./ILS-CSL
   ```

4. **Install the required packages**: Run the installation script with the following command:

   ```bash
   sh install.sh
   ```

5. **Run the main code**: Execute the primary script by running:

   ```bash
   python main.py
   ```

## Project Structure

Here's an explanation of key directories and files in this repository:

- `suppl/`: Supplementary results of the experiments in the paper.
- `BN-structure/`: Contains the true Bayesian networks used in the experiments.
- `data/`: Holds the utilized data in txt and csv formats, along with the score files, note that due to the limitation of upload size, we only keep data of Asia and Child for test.
  For other datasets, you can access the data provided in an open-source repository in https://github.com/andrewli77/MINOBS-anc/tree/master/data/csv, and use `calculate_score.py` to generate the local scores.
- `out/`: Stores the output and statistics from the experiments. The key results are illustrated as follows:
  - `adj-matrix/`: Contains the adjacency matrices of the learned structures with every iteration.
  - `gpt_history/`: Contains the history of GPT queries on all pairwise variables in the experiment.
  - `results-all.csv`: Contains all the related statistics of the experiment, including time, extra, missing, reverse, shd, norm_shd, nnz, fdr, tpr, fpr, precision, recall, F1, gscore, etc.
  - `llm_extimation.json`: Details of parameter estimation in Table 6 and Table 7.
  - `prior-iter`: Detailed data of prior constraints in the iterative process, as reported in Figure 6.
- `img/`: Contains the visual results generated from the experiments.
- `pairwise_samples/`: Includes the sampled pairs of nodes used for study.
- `perform/`: Contains the implemented or utilized prior constraint-based algorithms.
- `hc/`: Holds the extended HC search algorithm for prior constraints.
- `prompt/`: Contains meta materials used to generate prompts for LLMs.
- `minobsx/`: Contains extended MINOBSx search algorithm for soft constraints.
- `BI-CaMML/`: Contains CaMML; refer to `install.sh` for installation details. Requires a JRE environment for execution.

## Code Overview

- `main.py`: The main script for executing the ILS-CSL experiment.
- `config.py`: Manages the project configuration settings.
- `evaluation_DAG.py`: Responsible for the evaluation of Directed Acyclic Graphs (DAGs).
- `GPT.py`: Provides access to the GPT model.
- `statics_plot.py`: Includes scripts for generating experiment results and plotting figures.
- `graph_vis.py`: Used for visualizing the topological graph.
- `graph_sampler.py`: Samples pairs from the graph for study.
- `utils.py`: Contains utility functions for the project.
- `calculate_score.py`: Holds code for calculating scores such as BIC and BDeu. Not necessary once the score is calculated.
- `para_estimate.ipynb`: Estimate the parameters related to LLM inference and CSL structures in Section 4.2 and Appendix A.4.

## Running the Core Experiment

### Quick Start

To run the core ILS-CSL code, execute the following command:

```bash
python main.py --dataset=asia --alg=HC --score=bdeu --data_index=1 --datasize_index=0
```

This command saves the raw output in the `out/` folder and figures in the `img/` folder. Further process more statistics by running:

```bash
python statics_plot.py
```

### Configuration Options

- `dataset`: Specify the dataset name (e.g., asia, alarm, child, cancer, insurance, mildew, water, barley).
- `alg`: Define the algorithm name (e.g., CaMML, HC, softHC, hardMINOBSx, softMINOBSx).
- `score`: Indicate the score name (e.g., bic, bdeu).
- `data_index`: Set the index of the dataset (e.g., 1, 2, 3, 4, 5, 6).
- `datasize_index`: Set the index of the dataset size (0 for smaller size, 1 for larger size).
