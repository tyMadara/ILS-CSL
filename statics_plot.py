import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from rich import print as rprint
from scipy.stats import ttest_ind

from utils import *

# Configure the visual settings for seaborn
sns.set_theme(style="whitegrid")


def plot_metrics_with_iter(exp_name, result_data, metrics=None, img_dir="img/iter"):
    """
    Plot the metrics of the result.

    Args:
        exp_name: Name of the experiment.
        result_data: Dataframe with the metrics results.
        metrics: List of metrics to plot.
        img_dir: Directory to save the plot image.
    """

    plt.figure(figsize=(8, 6))

    if metrics is None:
        metrics = ["shd", "extra", "missing", "reverse"]

    markers = ['o', '^', 's', 'D', 'v', '*', '<', '>', 'P', 'X']
    colors = sns.color_palette("Set2")

    for i, metric in enumerate(metrics):
        sns.lineplot(data=result_data, x="iter_num", y=metric,
                     label=metric, color=colors[i % len(colors)],
                     marker=markers[i % len(markers)], markersize=10)

    plt.setp(plt.gca().lines, lw=2)
    plt.title(f"{exp_name.replace('-', ' ')} Metrics")
    plt.xlabel("Iteration")
    plt.ylabel("Metrics")
    plt.legend()
    plt.ylim(0, result_data["shd"].max())

    os.makedirs(img_dir, exist_ok=True)
    metrics_img_path = os.path.join(img_dir, f"{exp_name}.png")
    plt.savefig(metrics_img_path)
    rprint(
        f"📊 Metrics plot saved to {ColorP.path(metrics_img_path)}")
    plt.close()


def plot_metrics_with_LLM(exp_name, LLM_static):
    """
    Compare results with and without LLM, 
    and plot the SHD in a violin plot with statistical significance in the title.
    """
    plt.figure(figsize=(8, 6))

    t_stat, p_val = stats.ttest_rel(
        LLM_static['without LLM'], LLM_static['with LLM'])
    sns.violinplot(data=LLM_static,
                   palette=sns.color_palette("Set2"), linewidth=2)

    plt.title(f"{exp_name.replace('-', ' ')} SHD (p = {p_val:.3f})")
    plt.xlabel("with/without LLM")
    plt.ylabel("Metrics")

    metrics_img_path = f"img/LLM/{exp_name}.png"
    plt.savefig(metrics_img_path)
    rprint(
        f"🎻 LLM [yellow]violin[/yellow] plot saved to {ColorP.path(metrics_img_path)}")
    plt.close()


def is_empty_file(filepath):
    with open(filepath, "r") as f:
        lines = f.read()
        if len(lines.strip()) == 0:
            return True
        else:
            return False


@warning_filter
def plot_metrics_with_LLM_all(alg_name, score):
    """
    Compare results with and without LLM for all experiments, 
    and plot the SHD in a bar chart.
    """
    input_path = "out/result-all.csv"
    df = pd.read_csv(input_path)
    df = df[df["alg"] == alg_name]
    df = df[df["score"] == score]
    df = df[df["LLM"].isin(["with", "without"])]
    df["exp name"] = df.apply(
        lambda row: f"{row['d']}-{row['s']}", axis=1)

    # Add the results of the p-test here
    for exp_name in df["exp name"].unique():
        df_exp = df[df["exp name"] == exp_name]
        with_LLM = df_exp[df_exp["LLM"] == "with"]["shd"].values
        without_LLM = df_exp[df_exp["LLM"] == "without"]["shd"].values
        t_stat, p_val = ttest_ind(with_LLM, without_LLM)
        df.loc[df["exp name"] == exp_name, "p_val"] = p_val

    plt.figure()
    catplot = sns.catplot(data=df, x='exp name', y='shd', hue='LLM',
                          kind='bar', palette=sns.color_palette(),
                          legend_out=False, height=8, aspect=2.5)

    # Looping over the bars to add the values on top
    for i, bar in enumerate(catplot.ax.patches):
        catplot.ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{bar.get_height():.2f}',
                        ha='center', va='bottom')

    plt.xticks()
    catplot.set_axis_labels("Dataset", "SHD")
    # Add some space at the top for the p-values
    catplot.set(ylim=(0, df['shd'].max() + 0.1))
    catplot.fig.suptitle(f"{alg_name} {score} SHD")

    plt.legend()

    metrics_img_path = f"img/shd-{alg_name}-{score}.png"
    catplot.savefig(metrics_img_path)
    print(f"📊📊 LLM catplot saved to {metrics_img_path}")
    plt.close()


def plot_main(sim=False):
    for alg_name, score in alg_score_list:
        for dataset in dataset_list:
            for i, size_dict in enumerate(dataset2size):
                size = size_dict[dataset]
                LLM_static = []

                for data_index in data_index_list:
                    if sim and data_index < 20:
                        continue
                    if not sim and data_index >= 20:
                        continue
                    filepath = os.path.join(
                        result_dir, f"{dataset}-{size}-{data_index}-{alg_name}.csv")
                    if not os.path.exists(filepath):
                        rprint(
                            f"❓ {ColorP.path(filepath)} [red]not exists[/red]")
                        continue

                    # rprint(f"{filepath} [green]exists[/green]")
                    df = pd.read_csv(filepath)
                    LLM_static.append(
                        {"without LLM": df.iloc[0]["shd"], "with LLM": df.iloc[-1]["shd"], "exp name": f"{dataset}-{size}"})

                    # plot_metrics_with_iter(
                    #     f"{dataset}-{size}-{data_index}-{alg_name}", df)

                # df_LLM_static = pd.DataFrame(LLM_static)
                # if len(LLM_static) > 0:
                #     plot_metrics_with_LLM(
                #         f"{dataset}-{size}-{alg_name}", df_LLM_static[["without LLM", "with LLM"]])
                # else:
                #     rprint(
                #         f"📊 {ColorP.path({dataset}-{size}-{alg_name})} has no data")


def merge_all_result_csv():
    def iternum2status(iter_num, max_iter_num):
        if iter_num == 1:
            return "without"
        elif iter_num == max_iter_num:
            return "with"
        else:
            return "process"

    def get_num(d):
        correct_num, incorrect_num = 0, 0
        normal_num, exist_num, forb_num = 0, 0, 0
        if "edges" not in d:
            rprint("❓ edges not exists")
            return False
        for edge in d["edges"]:
            if edge["correct"] == "true":
                correct_num += 1
            else:
                incorrect_num += 1
            if edge["prior"] == "normal":
                normal_num += 1
            elif edge["prior"] == "exist":
                exist_num += 1
            elif edge["prior"] == "frob":
                forb_num += 1
            else:
                raise ValueError("edge prior error")
        return correct_num, incorrect_num, normal_num, exist_num, forb_num

    def load_true_edges(dataset):
        # load the true edges
        true_edges = []
        true_dag_path = f"BN_structure/{dataset}_graph.txt"
        true_dag = np.loadtxt(true_dag_path, dtype=int)
        for i, row in enumerate(true_dag):
            for j, value in enumerate(row):
                if value == 1:
                    true_edges.append([i, j])
        return true_edges

    def static_query(icsl, queried_edges, d):
        icsl.init_edge_prior()
        uncertain_query_num, correct_query_num, wrong_query_num = 0, 0, 0
        queried_edges.extend(d["edges"])
        for edge in queried_edges:
            edge_str = [icsl.i2p[edge['edge'][0]], icsl.i2p[edge['edge'][1]]]
            result = icsl.GPT_quiz(edge_str)
            icsl.add_prior(result)
            ans = result['answer']
            GT = icsl.GT_bot(edge_str)
            if ans == "D":
                uncertain_query_num += 1
            elif ans == GT:
                correct_query_num += 1
            else:
                wrong_query_num += 1
        return uncertain_query_num, correct_query_num, wrong_query_num

    rprint("🔀 Merging all result csv...")
    output_path = "out/result-all.csv"
    data_all = []
    for dataset in dataset_list:
        true_edges = load_true_edges(dataset)
        for datasize_index, size_dict in enumerate(dataset2size):
            size = size_dict[dataset]
            for data_index in data_index_list:
                for alg_name, score in alg_score_list:
                    # icsl = Iteration_CSL(
                    #     dataset, alg=alg_name, data_index=data_index, datasize_index=datasize_index, score=score)
                    # icsl.chatgpt = GPT(icsl.LLM_model)  # init the chatgpt
                    # icsl.edge_prior = EdgePrior(
                    #     icsl.true_dag, icsl.i2p, icsl.p2i)
                    # icsl.history = read_json(
                    #     icsl.history_path)  # load the history
                    exp_name = f"{dataset}-{size}-{data_index}-{alg_name}-{score}"
                    filepath = f"out/metrics/{exp_name}.csv"
                    filepath_prior_iter = f"out/prior-iter/{exp_name}.json"
                    if not os.path.exists(filepath):
                        rprint(
                            f"❓ {ColorP.path(filepath)} [red]not exists[/red]")
                        continue
                    elif not os.path.exists(filepath_prior_iter):
                        rprint(
                            f"❓ {ColorP.path(filepath_prior_iter)} [red]not exists[/red]")
                        continue
                    elif is_empty_file(filepath) or is_empty_file(filepath_prior_iter):
                        rprint(
                            f"❓ {ColorP.path(filepath)} [red]is empty[/red]")
                        continue
                    else:
                        rprint(
                            f"📊 {ColorP.path(filepath)} [green]exists[/green]")

                    df_metrics = pd.read_csv(filepath)
                    params = {"dataset": dataset, "s": size,
                              "r": data_index, "alg_name": alg_name, "score": score}
                    df_metrics = df_metrics.assign(**params)
                    cols = ["dataset", "s", "r", "alg_name", "score", "iter_num", "LLM", "time", "extra", "missing",
                            "reverse", "shd", "norm_shd", "nnz", "fdr", "tpr", "fpr", "precision", "recall", "F1", "gscore"]
                    df_metrics["iter_num"] = df_metrics["iter_num"].astype(int)
                    df_metrics["norm_shd"] = df_metrics["shd"] / \
                        len(true_edges)
                    max_iter_num = df_metrics["iter_num"].max()
                    if max_iter_num == 1:
                        df_metrics["LLM"] = "both"
                    else:
                        for index, row in df_metrics.iterrows():
                            df_metrics.at[index, "LLM"] = iternum2status(
                                row["iter_num"], max_iter_num)
                    data_metrics = json.loads(
                        df_metrics[cols].to_json(orient="records"))

                    data_prior_iter_raw = read_json(
                        filepath_prior_iter, quiet=True)
                    data_prior_iter = []
                    queried_edges = []
                    for iter, d in enumerate(data_prior_iter_raw):
                        correct_num, incorrect_num = 0, 0
                        normal_num, exist_num, forb_num = 0, 0, 0
                        nums = get_num(d)
                        if not nums:
                            continue
                        exist_prior_wrong_num, forb_prior_wrong_num = 0, 0
                        for edge in d["exist_edges"]:
                            if edge not in true_edges:
                                exist_prior_wrong_num += 1
                        for edge in d["forb_edges"]:
                            if edge in true_edges:
                                forb_prior_wrong_num += 1

                        correct_num, incorrect_num, normal_num, exist_num, forb_num = nums
                        uncertain_query_num, correct_query_num, wrong_query_num = 0, 0, 0
                        # uncertain_query_num, correct_query_num, wrong_query_num = static_query(
                        #     icsl, queried_edges, d)
                        rprint(
                            f"{dataset}-{size}-{data_index}-{alg_name}-{score}-iter{iter+1} exist_prior_wrong_num:{exist_prior_wrong_num} forb_prior_wrong_num:{forb_prior_wrong_num}")
                        static = {"dataset": dataset, "s": d["s"], "r": d["r"], "alg_name": alg_name, "score": score, "iter_num": iter + 1, "exist_prior": len(d["exist_edges"]), "forb_prior": len(d["forb_edges"]),
                                  "uncertain_query": uncertain_query_num, "correct_query": correct_query_num, "wrong_query": wrong_query_num,
                                  "exist_prior_wrong": exist_prior_wrong_num, "forb_prior_wrong": forb_prior_wrong_num,
                                  "ancs_prior": len(d["ancs"]), "edge": len(d["edges"]), "correct": correct_num, "incorrect": incorrect_num, "normal": normal_num, "exist": exist_num, "forb": forb_num}

                        data_prior_iter.append(static)
                    for i in range(len(data_metrics)):
                        for key in ["dataset", "s", "r", "alg_name", "score", "iter_num"]:
                            assert data_metrics[i][key] == data_prior_iter[i][key]
                    # merge metrics and prior_iter
                    for i in range(len(data_metrics)):
                        data_metrics[i].update(data_prior_iter[i])
                    data_all.extend(data_metrics)
    df = pd.DataFrame(data_all)
    df = df.rename(
        columns={"dataset": "d", "alg_name": "alg", "iter_num": "iter"})
    df.to_csv(output_path, index=False)
    rprint(f"✅ All result csv merged and saved to {ColorP.path(output_path)}")


def plot_iter_all(alg_name):
    input_path = "out/prior-iter-all.csv"
    df = pd.read_csv(input_path)
    df = df[df["alg_name"] == alg_name]
    df = df[df["LLM"].isin(["with", "without"])]
    df["exp name"] = df.apply(
        lambda row: f"{row['dataset']}-{row['s']}", axis=1)


def both2doublesample(df):
    """
    LLM = both -> two rows, one with LLM = with, one with LLM = without
    """
    df_both = df[df["LLM"] == "both"]
    df_both_with = df_both.copy()
    df_both_without = df_both.copy()
    df_both_with["LLM"] = "with"
    df_both_without["LLM"] = "without"
    df = pd.concat([df, df_both_with, df_both_without])
    df = df[df["LLM"].isin(["with", "without"])]
    return df


def output_table1(alg="CaMML", score="bdeu"):
    input_path = "out/result-all.csv"
    df = pd.read_csv(input_path)
    df = both2doublesample(df)
    df = df[df["alg"] == alg]
    df = df[df["score"] == score]

    df["exp name"] = df.apply(lambda row: f"{row['d']}-{row['s']}", axis=1)
    df = df[['exp name', 'LLM', 'norm_shd']]

    mean_df = df.groupby(["exp name", "LLM"]).mean().reset_index()
    mean_df = mean_df.pivot(index="exp name", columns="LLM", values="norm_shd")
    mean_df = mean_df.rename(columns={"without": "base", "with": "ILS-CSL"})
    std_df = df.groupby(["exp name", "LLM"]).std().reset_index()
    std_df = std_df.pivot(index="exp name", columns="LLM", values="norm_shd")
    std_df = std_df.rename(
        columns={"without": "base std", "with": "ILS-CSL std"})
    count_df = df.groupby(["exp name", "LLM"]).count().reset_index()
    count_df = count_df.pivot(
        index="exp name", columns="LLM", values="norm_shd")
    count_df = count_df.rename(
        columns={"without": "base num", "with": "ILS-CSL num"})

    final_df = pd.concat([mean_df, std_df, count_df], axis=1)

    final_df = final_df.round(4)
    final_df = final_df[['base', 'base std', 'base num',
                         'ILS-CSL', 'ILS-CSL std', 'ILS-CSL num']]
    alg_score = f"{alg}-{score}"

    final_df = final_df.transpose()

    final_df.to_csv(f"out/paper/table1-{alg_score}.csv")
    final_df.to_excel(f"out/paper/table1-{alg_score}.xlsx")


def merge_table1_topublish():
    def compute_improve(row):
        if row['base'] == 0:
            return "0\\%"
        else:
            improve = round(100*(row['ILS-CSL']-row['base'])/row['base'])
            if improve >= 0:
                return "\\textcolor{blue}"+"{+"+str(improve)+"\\%}"
            else:
                return "\\textcolor{red}"+"{"+str(improve)+"\\%}"

    def add_hline(filepath):
        string = read_txt(filepath).replace("\\\\", "\\")
        lines = string.split("\\\\\n")
        for i, line in enumerate(lines):
            if len(lines[i]) > 0:
                lines[i] += "\\\\"
                if i % 2 == 0:
                    lines[i] += "\\hline"
        string = "\n".join(lines)
        write_txt(filepath, string)
    df_all_latex = pd.DataFrame()
    for alg, score in alg_score_list:
        alg_score = f"{alg}-{score}"
        table1_path = f"out/paper/table1-{alg_score}.csv"
        df = pd.read_csv(table1_path, index_col=0)
        df = df.transpose().fillna(0.0).round(2)
        df[alg_score] = df.apply(
            lambda row: f"{row['base']:.2f}"+"{\\tiny ±"+f"{row['base std']:.2f}"+"}", axis=1)

        df["improve"] = df.apply(
            lambda row: compute_improve(row), axis=1)
        df[f"+ILS-CSL"] = df.apply(
            lambda row: f"{row['ILS-CSL']:.2f}"+"{\\tiny ±"+f"{row['ILS-CSL std']:.2f}"+"}\\raisebox{0.5ex}{\\tiny"+row['improve']+"}", axis=1)
        df = df[[alg_score, f"+ILS-CSL"]]
        df_all_latex = pd.concat([df_all_latex, df], axis=1)

    dataset_size = []
    for dataset in dataset_list[:4]:
        for dataset_index in range(2):
            dataset_size.append(
                f"{dataset}-{dataset2size[dataset_index][dataset]}")
    df_topublish = df_all_latex.loc[dataset_size].transpose()
    df_topublish.to_csv("out/table1.1.csv", sep="&",
                        lineterminator="\\\\\\\\\n", quoting=csv.QUOTE_NONE, escapechar='\\')
    add_hline("out/table1.1.csv")
    dataset_size = []
    for dataset in dataset_list[4:]:
        for dataset_index in range(2):
            dataset_size.append(
                f"{dataset}-{dataset2size[dataset_index][dataset]}")
    df_topublish = df_all_latex.loc[dataset_size].transpose()
    df_topublish.to_csv("out/table1.2.csv", sep="&",
                        lineterminator="\\\\\\\\\n", quoting=csv.QUOTE_NONE, escapechar='\\')
    add_hline("out/table1.2.csv")



def output_table3(alg, score):
    input_path = "out/result-all.csv"
    df = pd.read_csv(input_path)
    df = df[df["LLM"] == "with"]
    df = df[df["alg"] == alg]
    df = df[df["score"] == score]
    df["exp name"] = df.apply(lambda row: f"{row['d']}-{row['s']}", axis=1)
    df = df[['exp name', 'exist_prior', 'exist_prior_wrong',
             'forb_prior', 'forb_prior_wrong']]
    # mean, std, count
    mean_df = df.groupby(["exp name"]).mean().reset_index()
    mean_df = mean_df.rename(
        columns={"exist_prior": "exist prior num", "exist_prior_wrong": "wrong exist num", "forb_prior": "forb prior num", "forb_prior_wrong": "wrong forb num"})
    std_df = df.groupby(["exp name"]).std().reset_index()
    std_df = std_df.rename(
        columns={"exist_prior": "exist prior std", "forb_prior": "forb prior std"})

    mean_df = mean_df.set_index("exp name")
    std_df = std_df.set_index("exp name")
    final_df = pd.concat([mean_df, std_df], axis=1)
    final_df = final_df[["exist prior num", "exist prior std", "wrong exist num",
                        "forb prior num", "forb prior std", "wrong forb num"]]

    final_df = final_df.round(4)
    final_df = final_df.transpose()
    final_df.to_csv(f"out/paper/table3-{alg}-{score}.csv")
    final_df.to_excel(f"out/paper/table3-{alg}-{score}.xlsx")


def output_figure1():
    def plot_figure1(alg, score, df, png_name):
        df_melted = df[['exp_name', 'wrong_prior', 'wrong_query']].melt(
            id_vars='exp_name', var_name='metric', value_name='value')

        plt.figure()
        catplot = sns.catplot(x='exp_name', y='value', hue='metric', data=df_melted, kind='bar',
                                ci='sd', palette=sns.color_palette(), legend_out=False, height=5, aspect=3)

        # Looping over the bars to add the values on top
        for i, bar in enumerate(catplot.ax.patches):
            catplot.ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                            f'{bar.get_height():.2f}', ha='center', va='bottom')

        plt.xticks()
        catplot.set_axis_labels("Dataset", "num")
        # rotate x label
        catplot.set_xticklabels(rotation=15, ha='right')
        # Add some space at the top for the p-values
        # catplot.set(ylim=(0, df['shd'].max() + 0.1))
        catplot.fig.suptitle(f"{alg} {score} wrong number")

        plt.legend()
        catplot.savefig(png_name)
        plt.close()
        rprint(
            f"📊📊 LLM catplot saved to [italic blue]{png_name}[/italic blue]")

    for alg, score in alg_score_list:
        df_all = pd.DataFrame()
        for dataset in dataset_list:
            for i in range(2):
                datasize = dataset2size[i][dataset]
                input_path = "out/result-all.csv"
                df = pd.read_csv(input_path)
                df = df[df["alg"] == alg]
                df = df[df["score"] == score]
                df = df[df["d"] == dataset]
                df = df[df["s"] == datasize]
                df["exp name"] = df.apply(
                    lambda row: f"{row['d']}-{row['s']}", axis=1)
                df["exp name duo"] = df.apply(
                    lambda row: f"{row['d']}", axis=1)

                df = df[df["LLM"].isin(["with", "both"])]

                df = df[['exp name', "exp name duo", 'iter', 'exist_prior', 'exist_prior_wrong',
                        'forb_prior', 'forb_prior_wrong', 'uncertain_query', 'correct_query', 'wrong_query']]

                df["prior_wrong"] = df["exist_prior_wrong"] + \
                    df["forb_prior_wrong"]
                df = df[['exp name', "exp name duo",
                         'prior_wrong', 'wrong_query']]
                df = df.rename(
                    columns={"exp name": "exp_name", "exp name duo": "exp_name_duo",
                             "prior_wrong": "wrong_prior", "wrong_query": "wrong_query"})
                # df = df.set_index("exp name")
                df_all = pd.concat([df_all, df], axis=0)
        df_duo = df_all[['exp_name_duo', 'wrong_prior', 'wrong_query']].rename(
            columns={"exp_name_duo": "exp_name"})
        # save to csv
        df_all[['exp_name', 'wrong_prior', 'wrong_query']].to_csv(
            f"out/paper/figure1/figure1-{alg}-{score}.csv", index=False)
        df_duo.to_csv(
            f"out/paper/figure1/figure1-{alg}-{score}-duo.csv", index=False)
        # seaborn barplot
        plot_figure1(alg, score, df_all,  f"img/figure1/{alg}-{score}.png")
        plot_figure1(alg, score, df_duo,  f"img/figure1/{alg}-{score}-duo.png")


def output_figure2():
    for alg, score in alg_score_list:
        for dataset in dataset_list:
            input_path = "out/result-all.csv"
            df = pd.read_csv(input_path)
            df = df[df["alg"] == alg]
            df = df[df["score"] == score]
            df = df[df["d"] == dataset]

            df = df[['iter', 'exist_prior', 'exist_prior_wrong',
                    'forb_prior', 'forb_prior_wrong']]

            df["wrong_prior"] = df["exist_prior_wrong"] + \
                df["forb_prior_wrong"]
            df["prior_num"] = df["exist_prior"] + df["forb_prior"]

            # save to csv
            df = df[['iter', 'prior_num', 'wrong_prior']]
            df.to_csv(
                f"out/paper/figure2/figure2-{alg}-{score}-{dataset}.csv", index=False)
            # melt
            df_melted = df.melt(
                id_vars='iter', var_name='metric', value_name='value')
            sns.lineplot(x="iter", y="value", hue="metric",
                         data=df_melted, palette=sns.color_palette())
            plt.savefig(
                f"img/figure2/{alg}-{score}-{dataset}.png")
            plt.close()


def output_figure3():
    for alg, score in alg_score_list:
        for dataset in dataset_list:
            input_path = "out/result-all.csv"
            df = pd.read_csv(input_path)
            df = df[df["alg"] == alg]
            df = df[df["score"] == score]
            df = df[df["d"] == dataset]

            df = df[['iter', 'tpr', 'norm_shd']]
            df = df.rename(columns={"tpr": "TPR", "norm_shd": "SHD"})
            df["SHD"] = df["SHD"].round(4)
            df.to_csv(
                f"out/paper/figure3/figure3-{alg}-{score}-{dataset}.csv", index=False)
            # melt
            df_melted = df.melt(
                id_vars='iter', var_name='metric', value_name='value')
            sns.lineplot(x="iter", y="value", hue="metric",
                         data=df_melted, palette=sns.color_palette())
            plt.savefig(
                f"img/figure3/{alg}-{score}-{dataset}.png")
            plt.close()
            rprint(
                f"📊 Iter-tpr lineplot saved to [italic blue]{alg}-{score}-{dataset}.png[/italic blue]")


def prior_iter2adj_matrix():
    def edge_list2adj_matrix(edges, size):
        adj_matrix = np.zeros((size, size))
        for edge in edges:
            adj_matrix[edge[0], edge[1]] = 1
        return adj_matrix
    prior_iter_dir = "out/prior-iter"
    adj_matrix_dir = "out/adj-matrix"
    for alg_name, score in alg_score_list:
        for dataset in dataset_list:
            true_dag_path = f"BN_structure/{dataset}_graph.txt"
            true_dag = np.loadtxt(true_dag_path, dtype=int)
            for i, size_dict in enumerate(dataset2size):
                size = size_dict[dataset]
                for data_index in data_index_list:
                    exp_name = f"{dataset}-{size}-{data_index}-{alg_name}-{score}"
                    prior_iter_filepath = f"{prior_iter_dir}/{exp_name}.json"
                    data = read_json(prior_iter_filepath)
                    for i in range(len(data)):
                        edges = data[i]["edges"]
                        edges = [edge["edge"] for edge in edges]
                        exp_iter_name = f"{exp_name}-iter{i+1}"
                        adj_matrix_filepath = f"{adj_matrix_dir}/{exp_iter_name}.txt"
                        adj_matrix = edge_list2adj_matrix(
                            edges, true_dag.shape[0])
                        np.savetxt(adj_matrix_filepath, adj_matrix, fmt="%d")
                        rprint(
                            f"✅ {ColorP.path(adj_matrix_filepath)} saved")


if __name__ == '__main__':
    result_dir = "out/metrics"
    dataset2size = [
        {"asia": 250, "child": 500, "insurance": 500, "alarm": 1000,
            "cancer": 250, "mildew": 8000, "water": 1000, "barley": 2000},
        {"asia": 1000, "child": 2000, "insurance": 2000, "alarm": 4000,
            "cancer": 1000, "mildew": 32000, "water": 4000, "barley": 8000}
    ]
    data_index_list = [1, 2, 3, 4, 5, 6]
    alg_score_list = [["CaMML", "mml"],  ["HC", "bdeu"], ["softHC", "bdeu"], ["hardMINOBSx", "bdeu"], ["softMINOBSx", "bdeu"],
                      ["HC", "bic"], ["softHC", "bic"], ["hardMINOBSx", "bic"], ["softMINOBSx", "bic"]]

    dataset_list = ["asia", "child"]

    prior_iter2adj_matrix() # save the prior iter to adj matrix
    merge_all_result_csv() # merge all result csv
    for alg, score in alg_score_list:
        output_table1(alg, score) # output table1
        output_table3(alg, score) # output table3
        plot_metrics_with_LLM_all(alg, score) # plot metrics with LLM
    merge_table1_topublish() # merge table1 to the format to publish
    output_figure1() # output figure1
    output_figure2() # output figure2
    output_figure3() # output figure3
