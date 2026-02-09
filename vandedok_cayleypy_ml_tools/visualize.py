import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from .utils import get_bfs_xy
from matplotlib import colormaps

def draw_trajectories(graph, model, trajs_len, trajs_num):
    model.eval()

    X, y = graph.random_walks(width=trajs_num, length=trajs_len, mode="classic", nbt_history_depth=trajs_len)
    # success_counts = {name: 0 for name in predictors.keys()}
    y_model = model(X)

    cmap = colormaps["viridis"]

    X_trajs = X.view(trajs_len, trajs_num, -1)

    y_model_trajs = y_model.view(trajs_len, trajs_num)
    y_trajs = y.view(trajs_len, trajs_num)

    fig, ax = plt.subplots(figsize=(6,6))
    for traj_i in range(trajs_num):
        y_model_tr = y_model_trajs[:,traj_i].detach().cpu().numpy()
        y_tr = y_trajs[:,traj_i].detach().cpu().numpy()
        ax.plot(y_tr, y_model_tr, c=cmap(traj_i/trajs_num), alpha=0.5)
        ax.grid(True)
    y_lim = ax.get_ylim()
    ax.plot([0,y_lim[1]], [0,y_lim[1]], c="black", linewidth=1, linestyle='--', alpha=0.8)
    ax.set_ylim([0,y_lim[1]])
    return fig, ax


def draw_bfs_distros(
        graph,
        model,
        max_bfs_diameter,
        max_layer_size = 10**20,
        batch_size = 1014,
):
    ### Getting BFS dataset  -- we can use it to evaluate the distances on the states close to the start ###
    graph.free_memory()
    torch.cuda.empty_cache()
    # graph_bfs = CayleyGraph(graph_def, device="cpu", verbose=3)
    graph_bfs = graph


    bfs_result = graph_bfs.bfs(max_layer_size_to_store=max_layer_size, max_layer_size_to_explore=max_layer_size, max_diameter=max_bfs_diameter)
    graph_bfs.free_memory()
    X_bfs, y_bfs = get_bfs_xy(bfs_result)
    X = X_bfs.to(graph.device)
    y = y_bfs.to(graph.device)

    total_states = X.shape[0]

    y_model = []
    model.eval()

    for start_i in tqdm(range(0, X.shape[0], batch_size), desc="Getting model predictions for BFS states to visualize distributions"):
        end_i = min(start_i + batch_size, total_states)
        batch = X[start_i:end_i] 
        y_model.append(model(batch).detach().cpu())


    y_model = torch.cat(y_model).flatten()

    y_true = y
    ### Visualizing the distribution ###
    # -1 column corresponds to the states whos distances we don't know through BFS. In general we want it to be higher than other columns

    y_distro = pd.DataFrame.from_dict(
        {
            "true_distance": y_true.detach().cpu().numpy(),
            # "rw_distance": y.detach().cpu().numpy(),
            "model": y_model.detach().cpu().numpy(),
        }
    )
            
    y_distro_long = y_distro.melt(
        id_vars=["true_distance"],
        value_vars=[ "model"],
        var_name="prediction_type",
        value_name="distance"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxenplot(data=y_distro_long, x="true_distance", y="distance", hue="prediction_type", ax=ax)
    ax.set_yticks(np.arange(0, 101, 1), minor=False)
    ax.grid(which="major", axis='both', visible=True, alpha=0.5)
    ax.set_ylim(0, 21)

    points=np.arange(0,len(bfs_result.layer_sizes))

    ax.scatter(x=points, y=points, marker="x", s=100, linewidth=4,c="black", label="True distance")
    ax.legend(loc=(0.2,0.80))
    ax.set_title("Predicted distances distribution")
    return fig, ax


def draw_losses(exp_dir, smooth_ema_alpha=0.1):
     ### Visualizing the losses ###
    try:
        df_rw = pd.read_csv(exp_dir / "logs" / "csv"/ "rw_reg.csv")

        df_rw['train_loss_smooth'] = df_rw['train_loss'].ewm(alpha=smooth_ema_alpha).mean()
        df_rw['val_loss_smooth'] = df_rw['val_loss'].ewm(alpha=smooth_ema_alpha).mean()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(df_rw.index, df_rw["train_loss_smooth"], label="Train Loss", color="blue")
        ax.plot(df_rw.index, df_rw["val_loss_smooth"], label="Validation Loss", color="orange")
        ax.set_ylabel("loss")
        ax.set_title(f"RW reg loss, smoothed with EWMA(alpha={smooth_ema_alpha})")
        # ax.set_xticks([updates[1::2])
        ax.grid(True)
        ax.set_ylim(0, 0.5)

        fig_rw = fig
        ax_rw = ax
    except Exception as e:
        print(f"Could not plot RW reg losses due to: {e}")
        fig_rw = None
        ax_rw = None

    try:
        df_bellman = pd.read_csv(exp_dir / "logs" / "csv"/ "bellman.csv")
        df_bellman['train_loss_smooth'] = df_bellman['train_loss'].ewm(alpha=smooth_ema_alpha).mean()
        df_bellman['val_loss_smooth'] = df_bellman['val_loss'].ewm(alpha=smooth_ema_alpha).mean()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(df_bellman.index, df_bellman["train_loss_smooth"], label="Train Loss", color="blue")
        ax.plot(df_bellman.index, df_bellman["val_loss_smooth"], label="Validation Loss", color="orange")
        ax.set_ylabel("loss")
        ax.set_title(f"Bellman loss, smoothed with EWMA(alpha={smooth_ema_alpha})")
        # ax.set_xticks([updates[1::2])
        ax.grid(True)
        ax.set_ylim(0, 0.5)
        fig_bellman = fig
        ax_bellman = ax
    except Exception as e:
        print(f"Could not plot Bellman losses due to: {e}")
        fig_bellman = None
        ax_bellman = None

    return fig_rw, ax_rw, fig_bellman, ax_bellman

def draw_eval_figures(
        graph, 
        model, 
        max_bfs, 
        trajs_len, 
        trajs_num,
        exp_dir 
    ):

    
    output_dir = exp_dir / "figures"
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, _ = draw_trajectories(graph, model, trajs_len=trajs_len, trajs_num=trajs_num)
    fig.savefig(output_dir / "trajectories.png")

    fig, _ = draw_bfs_distros(graph, model, max_bfs)
    fig.savefig(output_dir / "bfs_distros.png")

    fig_rw, ax_rw, fig_bellman, ax_bellman = draw_losses(exp_dir)
    if fig_rw is not None:
        fig_rw.savefig(output_dir / "rw_reg_loss.png")
    if fig_bellman is not None:
        fig_bellman.savefig(output_dir / "bellman_loss.png")
    # df_losses = pd.read_csv(exp_dir / "logs" / "csv"/ "bellman.csv")
    # df