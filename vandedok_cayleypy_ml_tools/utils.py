from tqdm.auto import tqdm
import json
import torch


def write_json(path, obj):
    with open(path, "w+") as file:
        json.dump(obj, file, indent=4)


def read_json(path):
    with open(path, "r") as file:
        return json.load(file)


def write_txt(path, obj):
    assert type(obj) is str, "Only strings are accepted for txt writing"
    with open(path, "w+") as file:
        file.write(obj)


def read_txt(path):
    with open(path, "r") as file:
        return file.read()



def get_bfs_xy(bfs_result):
    X_bfs = []
    y_bfs = []
    for layer_i, layer in bfs_result.layers.items():

        X_bfs.append(layer.cpu())
        y_bfs.append(torch.ones(layer.shape[0], dtype=torch.int) * layer_i)
    X_bfs = torch.cat(X_bfs, dim=0)
    y_bfs = torch.cat(y_bfs, dim=0)
    return X_bfs, y_bfs


def get_rw_true_distances(X, X_bfs, y_bfs, batch_size=128):
    torch.cuda.empty_cache()
    # Batch matching X_enc to X_bfs_enc for efficiency
    y_true = []
    total_states = X.shape[0]  # Number of BFS states
    for start_i in tqdm(range(0, X.shape[0], batch_size)):
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

        del diff, matches, idxs, found
        torch.cuda.empty_cache()
    return torch.cat(y_true, dim=0).int()
