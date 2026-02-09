from cayleypy import CayleyGraph, Predictor
from tqdm.auto import tqdm
import numpy as np
from .cfg import SingleEvalCfg
from logging import getLogger

logger = getLogger()

def measure_success(
    graph,
    predictors,
    beam_width,
    rw_length,
    n_trials,
    max_steps,
    beam_mode="simple",
    bfs_result_for_mitm=None,
    verbose=False,
):
    success_counts = {name: 0 for name in predictors.keys()}

    X, y = graph.random_walks(width=n_trials, length=rw_length, mode="nbt")

    X_trajs = X.view(rw_length, n_trials, -1)
    if bfs_result_for_mitm is not None:
        logger.warning("BFS result for MITM is provided but supported -- ignoring it for now")
    # y_trajs = y.view(rw_length, n_trials)
    for trial_i in tqdm(range(n_trials), desc=f"RW length: {rw_length}, Beam width: {beam_width}, Max steps: {max_steps}"):

        start_state = X_trajs[-1, trial_i].cpu().numpy()
        for predictor_name, predictor in predictors.items():

            graph.free_memory()
            result = graph.beam_search(
                start_state=start_state,
                beam_width=beam_width,
                max_steps=max_steps,
                predictor=predictor,
                beam_mode=beam_mode,
                return_path=False,
            )
            success_counts[predictor_name] += result.path_found

            if verbose:
                path_found_str = "True  " if result.path_found else "False "
                output_str = f"{trial_i:>3}. Predictor: {predictor_name} | Beam search success: {path_found_str}"
                if result.path_found:
                    output_str += f"| path_length: {result.path_length}"
                else:
                    output_str += "|"
                tqdm.write(output_str)

    return {name: count for name, count in success_counts.items()}, {name: count / n_trials for name, count in success_counts.items()}


def evaluate(graph: CayleyGraph, eval_cfg: SingleEvalCfg, predictors: dict[str, Predictor] = {}, bfs_result_for_mitm=None, verbose=False):

    if type(eval_cfg) is SingleEvalCfg:
        eval_cfgs_list = [eval_cfg]
    else:
        eval_cfgs_list = eval_cfg


    # predictor = Predictor(graph, model)
    success_counts_all = []
    success_rates_all = []
    for single_eval_cfg in tqdm(eval_cfgs_list, desc="Running evals:", total=len(eval_cfgs_list), leave=False):

        success_counts = {}
        success_rates = {}
        success_counts_predictors, success_rates_predictors = measure_success(
            graph,
            predictors,
            single_eval_cfg.beam_width,
            single_eval_cfg.rw_length,
            single_eval_cfg.n_trials,
            single_eval_cfg.beam_max_steps,
            bfs_result_for_mitm=bfs_result_for_mitm if single_eval_cfg.beam_mitm else None,
            beam_mode=single_eval_cfg.beam_mode,
            verbose=verbose
        )

        for predictor_name in predictors.keys():
            if predictor_name not in success_counts:
                success_counts[predictor_name] = success_counts_predictors[predictor_name]
        for predictor_name in predictors.keys():
            if predictor_name not in success_rates:
                success_rates[predictor_name] = success_rates_predictors[predictor_name]
        success_counts_all.append(success_counts)
        success_rates_all.append(success_rates)
    return success_counts_all, success_rates_all
