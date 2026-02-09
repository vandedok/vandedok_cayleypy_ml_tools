from typing_extensions import Self
from typing import Literal
from pydantic import BaseModel, model_validator, ValidationError
from torch.optim import Adam, AdamW, SGD
from .losses import AssymMAELoss, AssymMSELoss
from .models import PermMLParams, PermMLP, StateDistanceTransformerParams, StateDistanceTransformer
### Config ###


SUPPORTED_MODELS= ["PermMLP", "StateDistanceTransformer"]

class ModelCfg(BaseModel):
    which: Literal["PermMLP", "StateDistanceTransformer"] = "PermMLP"
    # state_size: int = 88
    # state_vocab_size: int = 88
    y_norm: float  = 1
    params: PermMLParams|StateDistanceTransformerParams = PermMLParams()

    @model_validator(mode='after')
    def validate_torch_model(self) -> Self:
        if self.which not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {self.which}. Choose one of {SUPPORTED_MODELS}")
        if self.which == "PermMLP" and not isinstance(self.params, PermMLParams):
            raise ValueError(f"Expected PermMLParams for PermMLP, got: {type(self.params)}")
        elif self.which == "StateDistanceTransformer" and not isinstance(self.params, StateDistanceTransformerParams):
            raise ValueError(f"Expected StateDistanceTransformerParams for StateDistanceTransformer, got: {type(self.params)}")
            
        return self
    
    def get_model(self, state_size, state_vocab_size):

        if self.which == "PermMLP":
            return PermMLP(
                state_size=state_size,
                state_vocab_size=state_vocab_size,
                y_norm=self.y_norm,
                hidden_dims=self.params.hidden_dims,
                dropout=self.params.dropout,
                layer_norm=self.params.layer_norm
            )
        elif self.which == "StateDistanceTransformer":
            return StateDistanceTransformer(
                state_size=state_size,
                state_vocab_size=state_vocab_size,
                y_norm=self.y_norm,
                embed_dim=self.params.embed_dim,
                num_heads=self.params.num_heads,
                num_layers=self.params.num_layers,
                ff_dim=self.params.ff_dim,
                dropout=self.params.dropout,
            )
        else:
            raise ValueError(f"Unknown model: {self.which}. Choose one of {SUPPORTED_MODELS}")

class RandomWalksCfg(BaseModel):
    mode: Literal["classic", "bfs", "nbt", "classic_cpp"] = "nbt"
    length: int = 256
    nbt_history_depth: int = 512
    width: int = 1000


SUPPORTED_OPTIMIZERS = ["SGD", "Adam", "AdamW"]

class OptimizerCfg(BaseModel):
    which: Literal[*SUPPORTED_OPTIMIZERS] = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float =  0
    betas: list[float] = [0.9, 0.999]
    eps: float = 1e-08
    amsgrad: bool = False
    momentum: float = 0.
    dampening: float = 0.
    nesterov: bool = False

    def get_optimizer(self, model_params):
        if self.which == "Adam":
            return Adam(
                params=model_params,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
                eps=self.eps,
            )
        elif self.which == "AdamW":
            return AdamW(
                params=model_params,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
                eps=self.eps,
            )
        elif self.which == "SGD":
            return SGD(
                params=model_params,
                lr=self.learning_rate,
                momentum=self.momentum,
                nesterov=self.nesterov,
                dampening=self.dampening,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.which}. Choose one of: {SUPPORTED_OPTIMIZRERS}")


SUPPORTED_LOSSES = ["MAE", "MSE"]
class RegLossCfg(BaseModel):
    which: Literal[*SUPPORTED_LOSSES] = "MSE"
    assym_tau: float = 0.5
    pred_reg_coef: None| float = None
    pred_reg_target: None| float = None

    def get_loss_fn(self):
        if self.which == "MAE":
            return AssymMAELoss(
                tau=self.assym_tau,
                pred_reg_coef=self.pred_reg_coef,
                pred_reg_target=self.pred_reg_target,
            )
        elif self.which == "MSE":
            return AssymMSELoss(
                tau=self.assym_tau,
                pred_reg_coef=self.pred_reg_coef,
                pred_reg_target=self.pred_reg_target,
            )
        else:
            raise ValueError(f"Unsupported loss function: {self.which}. Choose one of: {SUPPORTED_LOSSES}")

class TrainRWReg(BaseModel):
    loss: RegLossCfg = RegLossCfg()
    optimizer: OptimizerCfg =OptimizerCfg()
    num_epochs: int = 30
    batch_size: int = 1024
    early_stop_patience: int = 10
    save_every: None | int = 500 
    save_best: None | int = 1
    s3_sync_every: None | int = None

# class TrainBfsReg(BaseModel):
#     batch_size: int = 1024
#     learning_rate: float = 0.0001
#     max_bfs_layer: int = 5
#     num_epochs: int = 30
#     early_stop_patience: int = 10
#     loss_cfg: RegLossCfg = RegLossCfg()

class TrainBellmanCfg(BaseModel):
    loss: RegLossCfg = RegLossCfg()
    optimizer: OptimizerCfg =OptimizerCfg()
    bellman_discount: float = 1.
    add_random_states: bool = False
    n_updates: int = 20
    epochs_per_update: int = 10
    bellman_batch_size: int = 1024
    in_update_batch_size: int = 1024
    in_update_early_stop_patience: int = 10
    global_early_stop_patience: int = 10
    bfs_for_boundary: int = 1
    save_every: None | int = 500 
    save_best: None | int = 1
    s3_sync_every: None | int = None


class TrainCfg(BaseModel):
    random_walks: RandomWalksCfg = RandomWalksCfg()
    val_ratio: float = 0.1
    rw_reg: TrainRWReg | None = TrainRWReg()
    # bfs_reg: TrainBfsReg = TrainBfsReg()
    bellman: TrainBellmanCfg| None = TrainBellmanCfg()


class SingleEvalCfg(BaseModel):
    n_trials: int = 20
    rw_mode: Literal["classic", "bfs", "nbt"] = "nbt"
    rw_length: int = 512
    beam_width: int = 2**10
    beam_max_steps: int = 50
    beam_mode: Literal["simple", "advanced"] = "simple"
    beam_mitm: bool = False



class LoggingCfg(BaseModel):
    py_logs: bool = True
    csv_logs: bool = True
    tb_logs: bool = True

class CayleyMLCfg(BaseModel):
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    eval: SingleEvalCfg | list[SingleEvalCfg] = SingleEvalCfg()
    logging: LoggingCfg = LoggingCfg()
