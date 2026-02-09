import time
import logging
import csv
from cayleypy import CayleyGraph
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
from .cfg import CayleyMLCfg
from .exp_saver import ExperimentSaver


from copy import deepcopy

logger = logging.getLogger()
logging.basicConfig(level=20)
EARLY_STOP_VERBOSE=False




def get_train_val(X, y, val_ratio: float = 0.1, stratify: bool = False):
    total_size = X.shape[0]
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    X_train, y_train, X_val, y_val = train_test_split(X, y, train_size=train_size, stratify=y.cpu().numpy() if stratify else None, shuffle=True)
    return X_train, y_train, X_val, y_val


def epoch_val(model: torch.nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, loss_fn: torch.nn.Module, batch_size: int):
    model.eval()
    total_val_loss = 0
    val_size = X_val.shape[0]
    with torch.no_grad():
        for start in range(0, val_size, batch_size):
            end = min(start + batch_size, val_size)
            xb = X_val[start:end]
            yb = y_val[start:end].float().squeeze()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            total_val_loss += loss.item() * xb.size(0)
    avg_val_loss = total_val_loss / val_size
    avg_val_loss = loss_fn.scale_loss_val(avg_val_loss, model.y_norm, val_size)
    return avg_val_loss


class MLLogger:

    def __init__(self, labels: list[str], logs_dir: str, py_logs: bool, tb_logs: bool, csv_logs: bool):
        self.logs_dir = Path(logs_dir)
        self.tb_logs = tb_logs
        self.csv_logs = csv_logs
        self.py_logs = py_logs
        self.labels = set(labels)
        self.steps = {x:0 for x in self.labels}
        self.logs_dir.mkdir()
        if self.csv_logs:
            self.csv_dir = self.logs_dir / "csv"
            self.csv_dir.mkdir()
            for label in self.labels:
                csv_path = self.get_csv_path(label)
                with open(csv_path, mode="w", newline="") as f: 
                    writer = csv.writer(f)
                    writer.writerow(["step", "train_loss", "val_loss"])   

        if self.tb_logs:
            self.tensorboard_dir = self.logs_dir / "tensorboard"
            self.tensorboard_dir.mkdir()
            self.tb_writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def get_csv_path(self, label):
        return self.csv_dir / f"{label}.csv"

    def log(self, train_value, val_value, label=""):
        assert label in self.labels, f"Logger didn't find label '{label}' in self.labels: {self.labels}"
        walltime = time.time()
        step_i = self.steps[label]

        if self.py_logs:
            logger.info(f"{label}: Step: {step_i} | Train: {train_value:.5f} | Val: {val_value:.5f}")

        if self.csv_logs:
            csv_path = self.get_csv_path(label)
            with open(csv_path, mode="a", newline="") as f: 
                writer = csv.writer(f)
                writer.writerow([step_i, train_value, val_value])  

        if self.tb_logs:
            self.tb_writer.add_scalars(label, {"train": train_value, "val": val_value}, global_step=step_i, walltime=walltime) 

        self.steps[label] += 1

@torch.compile
def regress_epoch_train(X_train, X_val, y_train, y_val, model, loss_fn, optimizer, epoch_i, batch_size, weights=None):

    train_size = len(X_train)
    model.train()
    y_train = y_train / model.y_norm
    # don't renormalize val -- the model in eval mode uses y_norm internally
    total_train_loss = 0
    weights_b = None
    for start in range(0, train_size, batch_size):
        end = min(start + batch_size, train_size)
        xb = X_train[start:end]
        yb = y_train[start:end].float().squeeze()
        if weights is not None:
            weights_b = weights[start:end].float().squeeze()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb, weights=weights_b)
        else:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)
    avg_train_loss = float(total_train_loss / train_size)

    if X_val is not None:
        avg_val_loss = float(epoch_val(model, X_val, y_val, loss_fn, batch_size))
    else:
        avg_val_loss = torch.nan

    avg_train_loss = loss_fn.scale_loss_train(avg_train_loss, float(model.y_norm), train_size)   

    return avg_train_loss, avg_val_loss


def train_rw_reg(cfg: CayleyMLCfg, graph: CayleyGraph, model: torch.nn.Module, exp_saver: ExperimentSaver):
    loss_fn = cfg.train.rw_reg.loss.get_loss_fn()
    optimizer  = cfg.train.rw_reg.optimizer.get_optimizer(model.parameters())
    rw_cfg = cfg.train.random_walks
    early_stopper = EarlyStopper(patience=cfg.train.rw_reg.early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, in_mem_saving=True, label="RW-reg", trace_func=logger.info)
    ml_logger = MLLogger(logs_dir=exp_saver.exp_dir/"logs", labels=["rw_reg"], py_logs=cfg.logging.py_logs, tb_logs=cfg.logging.tb_logs, csv_logs=cfg.logging.csv_logs)
    
    for epoch_i in range(cfg.train.rw_reg.num_epochs):
        X, y = graph.random_walks(width=rw_cfg.width, length=rw_cfg.length, mode=rw_cfg.mode, nbt_history_depth=rw_cfg.nbt_history_depth)
        X_train, X_val, y_train, y_val = get_train_val(X, y, cfg.train.val_ratio, stratify=True)
        avg_train_loss, avg_val_loss = regress_epoch_train(X_train, X_val, y_train, y_val, model, loss_fn, optimizer, epoch_i, cfg.train.rw_reg.batch_size)
        # logger.info(f"Epoch {epoch_i} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        ml_logger.log(train_value=avg_train_loss, val_value=avg_val_loss, label="rw_reg")
        early_stopper(avg_val_loss, model)

        exp_saver.trigger_saving_every("rw_reg_checkpoints", f"epoch_{epoch_i}_val_{avg_val_loss:.5f}", model, optimizer, epoch_i, cfg.train.rw_reg.save_every)
        exp_saver.trigger_saving_best("special_checkpoints", f"rw_reg_best_val_loss_{avg_val_loss:.5f}", model, optimizer, epoch_i, early_stopper.improvement_on_last_epoch, cfg.train.rw_reg.save_best)

        if early_stopper.early_stop:
            break
    
    exp_saver.save_training_state("special_checkpoints", f"rw_reg_last", model, optimizer, epoch_i)


# def train_bfs_reg(cfg: CayleyMLCfg, model, X_bfs, y_bfs):
#     loss_fn = cfg.train.bfs_reg.loss.get_loss_fn()
#     optimizer  = cfg.train.bfs_reg.optimizer.get_optimizer(model.parameters())

#     # X_train, X_val, y_train, y_val = get_train_val(X_bfs, y_bfs, cfg.train.val_ratio, stratify=False)
#     early_stopper = EarlyStopper(patience=bfs_cfg.early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, in_mem_saving=True, trace_func=logger.info)
#     model.train()

#     # Using weights as bfs dataset is heavily imbalanced
#     _, counts = y_bfs.unique(return_counts=True)
#     total = torch.sum(counts)
#     repeats = (total/counts).int()
#     weights = torch.pow(repeats, 0.5)[y_bfs]

#     for epoch_i in range(bfs_cfg.num_epochs):       
#         avg_train_loss, _ = regress_epoch_train(X_bfs, None, y_bfs, None, model, loss_fn, optimizer, epoch_i, bfs_cfg.batch_size, weights=weights)
#         logger.info(f"Epoch {epoch_i} | Train Loss: {avg_train_loss:.5f}")
#         early_stopper(avg_train_loss, model)
#         if early_stopper.early_stop:
#             model.load_state_dict(early_stopper.get_model_weights())
#             break

def get_neighbors(X, gens):
    n_actions = gens.shape[0]
    n_states = X.shape[0]
    expanded_states = X.unsqueeze(1).expand(-1, n_actions, -1)  # [N, A, S]
    expanded_gens = gens.unsqueeze(0).expand(n_states, -1, -1)  # [N, A, S]
    neighbors = torch.gather(expanded_states, 2, expanded_gens)  # [N, A, S]
    return neighbors


def get_true_distances(X, X_bfs, y_bfs, batch_size=128):
    # Batch matching X_enc to X_bfs_enc for efficiency
    y_true = []
    total_states = X.shape[0]  # Number of BFS states
    for start_i in range(0, X.shape[0], batch_size):
        end_i = min(start_i + batch_size, total_states)
        batch = X[start_i:end_i]

        diff = batch.unsqueeze(1) - X_bfs.unsqueeze(0)
        matches = (diff == 0).all(dim=2)  # [B, N]
        # For each batch element, find the first match (if any)
        idxs = matches.float().argmax(dim=1)
        found = matches.any(dim=1)
        # Assign y_bfs[idx] if found, else -1
        y_true_batch = torch.where(found, y_bfs[idxs], torch.full_like(idxs, -1))
        y_true.append(y_true_batch)
  
    return torch.cat(y_true, dim=0).int()

def bellman_update(model, X, y, generators, discount, X_bfs, y_bfs, batch_size, boundary_batch_size):
    model.eval()

    y_true = get_true_distances(X, X_bfs, y_bfs, batch_size=boundary_batch_size).float()
    flag_known = y_true!=-1
    X_to_find = X[~flag_known]
    y_to_find = y[~flag_known]

    X_neighbors = get_neighbors(X_to_find, generators)  # [Nf, A, S]
    X_neighbors_flat = X_neighbors.view(-1, X.shape[1])  # [Nf * A, S]
    with torch.no_grad():
        num_states = X_neighbors_flat.shape[0]
        preds_flat = torch.zeros(num_states, device=X_neighbors_flat.device)
        with torch.no_grad():
            for start in range(0, num_states, batch_size):
                end = min(start + batch_size, num_states)
                batch = X_neighbors_flat[start:end]
                preds_flat[start:end] = model(batch).squeeze()
        preds = preds_flat.view(X_neighbors.shape[0], -1)  # [Nf, A]

        targets = 1 + discount*preds.min(dim=1)[0]  # [Nf]

        targets = torch.min(targets, y_to_find)
        targets = torch.clamp_min(targets, 1)

        y_final = torch.empty_like(y_true)
        y_final[flag_known] = y_true[flag_known]
        y_final[~flag_known] = targets

    return y_final.detach()




def train_with_bellman(cfg: CayleyMLCfg, graph: CayleyGraph, model: torch.nn.Module, X_bfs: torch.Tensor , y_bfs: torch.Tensor, exp_saver: ExperimentSaver):
    
    loss_fn = cfg.train.bellman.loss.get_loss_fn()
    optimizer  = cfg.train.bellman.optimizer.get_optimizer(model.parameters())

    rw_cfg = cfg.train.random_walks
    global_early_stopper = EarlyStopper(patience=cfg.train.bellman.global_early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, in_mem_saving=True, label="Global-Bellman", trace_func=logger.info)
    in_update_early_stopper = EarlyStopper(patience=cfg.train.bellman.in_update_early_stop_patience, verbose=EARLY_STOP_VERBOSE, path=None, in_mem_saving=True, label="InUpdate-Bellman", trace_func=logger.info)
    epochs_per_update = cfg.train.bellman.epochs_per_update
    ml_logger = MLLogger(logs_dir=exp_saver.exp_dir/"logs", labels=["bellman", "bellman_in_update"], py_logs=cfg.logging.py_logs, tb_logs=cfg.logging.tb_logs, csv_logs=cfg.logging.csv_logs)

    for bellman_i in range(cfg.train.bellman.n_updates):
        torch.cuda.empty_cache()
        graph.free_memory()
        X, y = graph.random_walks(width=rw_cfg.width, length=rw_cfg.length, mode=rw_cfg.mode, nbt_history_depth=rw_cfg.nbt_history_depth)
        
        if cfg.train.bellman.add_random_states:
            X_train_rand = torch.argsort(torch.rand(X.shape[0], X.shape[1], device=graph.device), dim=1)
            X = torch.cat((X, X_train_rand))
            y = torch.cat((y, torch.full((X_train_rand.size(0),), float('inf'), device=graph.device)))
        
        generators = torch.tensor(graph.generators, device=graph.device)
        y_bellman = bellman_update(
            model=model, 
            X=X, 
            y=y,
            generators=generators,
            discount=cfg.train.bellman.bellman_discount,
            X_bfs=X_bfs,
            y_bfs=y_bfs,
            batch_size=cfg.train.bellman.bellman_batch_size, 
            boundary_batch_size=64,
            )  # In some setting goal state is not central state
        
        X_train, X_val, y_train, y_val = get_train_val(X, y_bellman, cfg.train.val_ratio, stratify=False)

        avg_train_loss_per_update = 0
        avg_val_loss_per_update = 0
        optimizer.state.clear() # clears accumulated values of the optimizer -- not sure if it is a good idea though
        in_update_early_stopper.reset()
        for epoch_i in range(epochs_per_update):
            
            avg_train_loss, avg_val_loss = regress_epoch_train(X_train, X_val, y_train, y_val, model, loss_fn, optimizer, epoch_i, cfg.train.bellman.in_update_batch_size)

            avg_train_loss_per_update += float(avg_train_loss)
            avg_val_loss_per_update += float(avg_val_loss)
            ml_logger.log(train_value=avg_train_loss, val_value=avg_val_loss, label="bellman_in_update")

            in_update_early_stopper(avg_val_loss, model)
            if in_update_early_stopper.early_stop:
                model.load_state_dict(in_update_early_stopper.get_model_weights())
                break

        avg_train_loss_per_update /= (epoch_i+1)
        avg_val_loss_per_update /= (epoch_i+1)
        global_early_stopper(avg_val_loss_per_update, model)

        exp_saver.trigger_saving_every("bellman_checkpoints", f"update_{bellman_i}_val_loss_{avg_val_loss:.5f}", model, optimizer, bellman_i, cfg.train.bellman.save_every)
        exp_saver.trigger_saving_best("special_checkpoints", f"bellman_best_update_{bellman_i}_val_loss_{avg_val_loss:.5f}", model, optimizer, bellman_i, global_early_stopper.improvement_on_last_epoch, cfg.train.bellman.save_best)
        exp_saver.trigger_s3_sync(bellman_i, cfg.train.bellman.s3_sync_every)
        if global_early_stopper.early_stop:
            break
        
        ml_logger.log(train_value=avg_train_loss, val_value=avg_val_loss, label="bellman")
        # logger.info(f"Bellman_update: {bellman_i} | Avg Train Loss: {avg_train_loss_per_update:.4f} | Avg Val Loss: {avg_val_loss_per_update:.4f}")

    exp_saver.save_training_state("special_checkpoints", f"bellman_last", model, optimizer, bellman_i)
    exp_saver.sync_with_s3()


### Taken and modified from  https://github.com/Bjarten/early-stopping-pytorch
#  commit fbd87a6135820700e27cda3448d5d54dc6fd3b0c
# MIT license

class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', in_mem_saving=False, label="", trace_func=logger.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to. Pass None to avoid saving.
                            Default: 'checkpoint.pt'
            in_mem_saving (bool): save model's weights on RAM
                            Default: False
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.in_mem_saving = in_mem_saving
        self.trace_func = trace_func
        self.label = label
        self.reset()

    def reset(self):
        self.early_stop = False
        self.best_val_loss = None
        self.counter = 0
        self.val_loss_min = np.inf
        self.model_state_dict = None
        self.improvement_on_last_epoch = False

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.improvement_on_last_epoch = True
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
            
        else:
            self.improvement_on_last_epoch = False # nneeds to be here to trace best model
            if self.patience < 0:
                pass
            else:
                # No significant improvement
                self.counter += 1
                self.trace_func(f'{self.label}-EarlyStopper counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, val_loss, model: torch.nn.Module):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')

        if self.in_mem_saving:
            # deepcopy keeps the state dict on the same device -- bad for memory usage, but probably good for speed
            self.model_state_dict = deepcopy(model.state_dict())
        if self.path is not None:
            if self.verbose:
                self.trace_func("Saving model ...")
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def get_model_weights(self):
        if not self.in_mem_saving:
            raise ValueError(" With in_mem_saving==False model's weights are not saved in RAM")
        return self.model_state_dict
    
