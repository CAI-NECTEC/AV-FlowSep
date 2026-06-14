import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from src import sampling
from src.odes import ODERegistry
from src.backbones import BackboneRegistry
from src.utils.inference import evaluate_model
import numpy as np

class AVFlowModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "avss_dit",
        ode: str = "flowmatching",
        lr: float = 1e-4,
        ema_decay: float = 0.999,
        t_eps: float = 0.03,
        num_eval_files: int = 50,
        loss_type: str = "mse",
        vocoder_path: str = None,
        data_module_cls = None,
        **kwargs
    ):
        super().__init__()

        # Init Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        
        # Init ODE
        ode_cls = ODERegistry.get_by_name(ode)
        self.ode = ode_cls(**kwargs)   

        # Hyperparams
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.vocoder_path = vocoder_path
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

        if kwargs.get("pretrained_talknet", None):
            print("freeze visual encoder")
            for p in self.dnn.visual_encoder.parameters():
                p.requires_grad = False
            self.dnn.visual_encoder.eval()

        torch.set_float32_matmul_precision("medium")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())      

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()    

    def train(self, mode=True, no_ema=False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, vectorfield, condVF):    
        if self.loss_type == "mse":
            err = vectorfield - condVF
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            err = vectorfield - condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss    

    def _step(self, batch):
        x, y, visual_features = batch
        rdm = (1 - torch.rand(x.shape[0], device=x.device)) * (self.ode.T - self.t_eps) + self.t_eps   
        t = torch.min(rdm, torch.tensor(self.ode.T))
        mean, std = self.ode.marginal_prob(x, t, y)
        z = torch.randn_like(x)
        sigmas = std[:, None, None, None]  
        xt = mean + sigmas * z
        der_mean = self.ode.der_mean(x, t, y)
        der_std = self.ode.der_std(t)
        condVF = der_std * z + der_mean
        vectorfield = self(xt, t, y, visual_features)
        loss = self._loss(vectorfield, condVF)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx): 
        loss = self._step(batch)
        # self.log('valid_loss', loss, on_step=False, on_epoch=True)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, self.vocoder_path)
            # self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('pesq', pesq, on_step=False, on_epoch=True, sync_dist=True)
            # self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True, sync_dist=True)
            # self.log('estoi', estoi, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def forward(self, x, t, y, context):
        dnn_input = torch.cat([x, y], dim=1)
        score = -self.dnn(dnn_input, t, context)
        return score    

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)     

    def get_ode_sampler(
        self, odesolver_name, y, context, N=None, **kwargs
    ): 
        N = self.ode.N if N is None else N
        ode = self.ode.copy()
        ode.N = N
    
        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_white_box_solver(
            odesolver_name, ode=ode, VF_fn=self, y=y, context=context, **kwargs
        )

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)
