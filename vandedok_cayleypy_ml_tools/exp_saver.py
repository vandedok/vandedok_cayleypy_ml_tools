
import logging
from pathlib import Path
from shutil  import rmtree
import torch
from .cfg import CayleyMLCfg
from .utils import write_json, read_json
from .s3 import sync_s3_bucket, get_r2_bucket_usage_with_api



class ExperimentSaver:

    def __init__(
            self, 
            exp_dir: None|str|Path, 
            cfg: CayleyMLCfg, 
            secrets:str|Path=None, 
            r2_limit_gb=8, 
            s3_sync_checkpoints=False,
            trace_func=print, 
            overwrite=False,
        ):
        self.exp_dir = Path(exp_dir)
        self.trace_func = trace_func

        if overwrite and self.exp_dir.exists():
            rmtree(self.exp_dir)
            
        if not self.exp_dir.exists():
            self.exp_dir.mkdir()
        else:
            raise ValueError("Experiment dir: {self.exp_dir} already exists, aborting")
        
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.best_models_dir = self.exp_dir / "best_model"
        self.best_cpt_path = None
        self.secrets = read_json(secrets) if secrets is not None else None
        self.r2_limit_gb = r2_limit_gb
        self.s3_sync_checkpoints = s3_sync_checkpoints
        self.save_cfg(cfg)
        

    def save_cfg(self, cfg: CayleyMLCfg|dict|str|Path):
        if type(cfg) is CayleyMLCfg:
            write_json(self.exp_dir/"cfg.json", cfg.model_dump())
        elif type(cfg) is str or isinstance(cfg, Path):
            cfg_dict = read_json(cfg)
            write_json(self.exp_dir/"cfg.json", cfg_dict)
        else:
            write_json(self.exp_dir/"cfg.json", cfg)

    def get_file_handler(self):
        log_file_path = self.exp_dir / "logs.txt"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        return file_handler
  

    def get_checkpoint_path(self, subpath, filename):
        path_to_return = self.exp_dir
        if subpath is not None:
            path_to_return = path_to_return / subpath
            path_to_return.mkdir(parents=True, exist_ok=True)
        if not filename.endswith("ckpt.pt") and not filename.endswith(".pt"):
            filename = f"{filename}.ckpt.pt"
        return path_to_return / filename
    
    def save_training_state(self, subpath, filename, model, optimizer, step_i):
        checkpoint = { 
            'step_i': step_i,
            'model': model.state_dict() if model is not None else None,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
        }
        if self.exp_dir is not None:
            path_to_save = self.get_checkpoint_path(subpath, filename)
            self.trace_func(f"Saving model to: {path_to_save}")
            torch.save(checkpoint, path_to_save) 

    def trigger_saving_every(self, subpath, filename, model, optimizer, step_i, save_every):
        if step_i!=0 and save_every is not None and step_i%save_every==0:
            self.save_training_state(subpath, filename, model, optimizer, step_i)

    def trigger_saving_best(self, subpath, filename, model, optimizer, step_i, is_best, save_best):
        # TODO: add saving multiple best models
        if is_best and save_best:
            if self.best_cpt_path is not None:
                self.best_cpt_path.unlink()
            self.save_training_state(subpath, filename, model, optimizer, step_i)
            self.best_cpt_path = self.get_checkpoint_path(subpath, filename)

    def trigger_s3_sync(self, step_i, sync_every):
        if self.secrets is not None:
            if sync_every is not None and step_i%sync_every==0:
                self.sync_with_s3()
        else:
            self.trace_func("No S3 secrets provided, skipping S3 sync.")

    def trigger_final_s3_sync(self, model):
        # Not saveing optimizer state for final weights to save space
        self.save_training_state("final_weights", "final_weights.pt", model, None, None)
        if self.secrets is not None:
            self.sync_with_s3()
        else:
            self.trace_func("No S3 secrets provided, skipping S3 sync.")

    def sync_with_s3(self, extra_exclude_patterns: list[str]=[]):
        if self.secrets is not None:
            try:
                do_sync=True
                if self.r2_limit_gb:
                    result = get_r2_bucket_usage_with_api(
                        account_id=self.secrets['CLOUDFLARE_ACCOUNT_ID'],
                        bucket_name=self.secrets['CLOUDFLARE_R2_BUCKET_NAME'],
                        api_token=self.secrets['CLOUDFLARE_R2_READ_TOKEN'],
                    )
                    
                    if int(result["payloadSize"]) + int(result["metadataSize"]) > self.r2_limit_gb * 2**30:
                        self.trace_func(f"S3 bucket size limit exceeded ({self.r2_limit_gb} GB), not syncing.")
                        do_sync = False

                if do_sync:
                    self.trace_func("Syncing experiment directory with S3...")

                    if self.s3_sync_checkpoints is False:
                        exclude_patterns = extra_exclude_patterns + ["*.ckpt.pt"]

                    sync_s3_bucket(
                        s3_endpoint=self.secrets['AWS_ENDPOINT_URL'],
                        s3_provider=self.secrets['AWS_PROVIDER'],
                        s3_access_key=self.secrets['AWS_ACCESS_KEY_ID'],
                        s3_secret_access_key=self.secrets['AWS_SECRET_ACCESS_KEY'],
                        path_to_dir=self.exp_dir,
                        bucket_name=self.secrets['AWS_BUCKET_NAME'],
                        exclude_patterns=exclude_patterns,
                    )

                    self.trace_func("Done.")
                
            except Exception as e:
                self.trace_func(f"Error syncing with S3: {e}")
