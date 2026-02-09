
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from argparse import ArgumentParser
import logging
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from cayleypy import  CayleyGraph, Predictor, CayleyGraphDef
from .utils import write_json, read_json, get_bfs_xy
from .eval import evaluate
from .visualize import draw_eval_figures
from .train import ExperimentSaver, train_rw_reg, train_with_bellman
from .cfg import CayleyMLCfg



logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--inputs_dir", "-i", type=Path, required=True, help="Directory containing input data.")
    parser.add_argument("--exp_out_dir", "-o", type=Path, required=True, help="Directory to save experiment outputs.")
    parser.add_argument("--cfg", "-c", type=Path, required=True, help="Path to the configuration file.")
    parser.add_argument("--puzzle_name", default="", type=str, required=False, help="Name of the puzzle.")
    parser.add_argument("--secrets", default=None, help="Path to the JSON file containing secrets (e.g. access tokens).")
    parser.add_argument("--overwrite", action='store_true', help="Whether to overwrite existing experiment outputs.")
    args = parser.parse_args()


    logger.info("Preparing experiment saver.")
    exp_saver = ExperimentSaver(args.exp_out_dir, cfg=args.cfg, overwrite=args.overwrite, trace_func=logger.info, secrets=args.secrets)
    logger.addHandler(exp_saver.get_file_handler())
    # logger.addHandler(logging.StreamHandler())
    logger.info("Now logs will be also saved to file.")

    try:
        logger.info("Loading puzzle info...")
        puzzle_info = read_json(args.inputs_dir / "puzzle_info.json")
        central_state = np.array(puzzle_info["central_state"])
        generators = {k: np.array(v) for k, v in puzzle_info["generators"].items()}

        logger.info("Central state %s", str(central_state))
        logger.info("Generators names: %s", str(list(generators.keys())))

        df_test = pd.read_csv(args.inputs_dir / "test.csv", index_col = "initial_state_id")
        df_sample_submission = pd.read_csv(args.inputs_dir / "sample_submission.csv", index_col = "initial_state_id")
        logger.info("Done.")


        logger.info("Setting up Cayley graph...")
        gens_names = list(generators.keys())
        graph_def = CayleyGraphDef.create(
            generators = [generators[x] for x in gens_names],
            generator_names = gens_names,
            central_state = central_state
        )
        graph = CayleyGraph(graph_def)
        device = graph.device
        logger.info("Done.")

        logger.info("Loading config...")
        cfg = read_json(args.cfg)
        cfg = CayleyMLCfg.model_validate(cfg)
        logger.info(cfg.model_dump())
        logger.info("Done.")

        logger.info("Setting up model...")
        state_destination = graph.central_state
        state_size =  len( graph.central_state )
        state_vocab_size = len(torch.unique(state_destination ))
        model = cfg.model.get_model(state_size,state_vocab_size)
        model.to(device)
        graph.free_memory()
        torch.cuda.empty_cache()
        logger.info("Done.")

        logger.info("Preparing BFS data...")
        graph_bfs = graph
        max_layer_size = 10**20
        max_diameter = cfg.train.bellman.bfs_for_boundary
        bfs_result = graph_bfs.bfs(max_layer_size_to_store=max_layer_size, max_layer_size_to_explore=max_layer_size, max_diameter=max_diameter)
        graph_bfs.free_memory()
        logger.info(f"Layers in BFS: {bfs_result.layers.keys()}")
        X_bfs, y_bfs = get_bfs_xy(bfs_result)
        X_bfs = X_bfs.to(device)
        y_bfs = y_bfs.to(device)
        logger.info("Done.")

        logger.info("Starting training...")
        if cfg.train.rw_reg is not None:
            train_rw_reg(cfg, graph, model, exp_saver)
        if cfg.train.bellman is not None:
            train_with_bellman(cfg, graph, model, X_bfs, y_bfs, exp_saver)
        logger.info("Training finished.")

        logger.info("Saving final model weights and syncing with S3 (if secrets are provided)...")
        exp_saver.save_training_state("weights", "final_weights.pt", model, None, None)
        exp_saver.sync_with_s3()
        logger.info("Done.")

        logger.info("Starting evaluation...")
        model.eval()
        model.set_cayleypy_inference(True)
        success_counts, success_rates = evaluate(
            graph, 
            cfg.eval,
            predictors={"model": Predictor(graph, model)}, 
            verbose=True
        )

        eval_results = {"success_counts": success_counts, "success_rates": success_rates}
        write_json(args.exp_out_dir/"eval.json",eval_results)

        print("Evaluation results:")
        print(eval_results)
        print("Evaluation finished.")

        logger.info("Drawing evaluation figures...")
        draw_eval_figures(
            graph, 
            model, 
            max_bfs=6, 
            trajs_len=100, 
            trajs_num=20,
            exp_dir=args.exp_out_dir
        )
        logger.info("Drawing finished.")

        logger.info("Final sync with S3 (if secrets are provided)...")
        exp_saver.sync_with_s3()
        logger.info("All done.")

    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        raise e
