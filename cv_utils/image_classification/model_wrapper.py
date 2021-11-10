import torch
import pytorch_lightning as pl

class CommonModelWrapper(pl.LightningModule):
    def __init__(self, model, loss, metrics, train_mode_in_inference=False):
        super().__init__()
        
        self.model = model
        self.loss = loss
        self.metrics = metrics
        
        self.optim_alg = torch.optim.AdamW
        self.optim_kwargs = {'lr':1e-4, 'weight_decay':2e-5, 'amsgrad':True}
        
        self.train_mode_in_inference = train_mode_in_inference
    
    def forward(self, x):
        return self.model.forward(x)
    
    def configure_optimizers(self):
        optimizer = self.optim_alg(self.parameters(), **self.optim_kwargs)
        return optimizer
    
    def training_step(self, batch, batch_nb):
        img, mask = batch
        
        out = self.forward(img)        
        loss = self.loss(out, mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            for name, func in self.metrics.items():
                self.log(f"train_{name}", func(out, mask), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_nb):
        img, mask = batch
        
        if self.train_mode_in_inference:
            self.model.train()
        
        out = self.forward(img)
        loss = self.loss(out, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            for name, func in self.metrics.items():
                self.log(f"val_{name}", func(out, mask), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_nb):
        img, mask = batch
        
        if self.train_mode_in_inference:
            self.model.train()
        
        out = self.forward(img)
        loss = self.loss(out, mask)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            for name, func in self.metrics.items():
                self.log(f"test_{name}", func(out, mask), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def load_model(self, checkpoint_path, evaluation=True, requires_grad=True):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.requires_grad_(requires_grad)
        
        if evaluation: self.model.eval()
        else: self.model.train()
    
    def cpu(self):
        self.model.cpu()
    
    def cuda(self):
        self.model.cuda()
    
    def set_optim_hyp(self, class_alg, hyp_dict):
        self.optim_alg = class_alg
        for key, val in hyp_dict.items():
            self.optim_kwargs[key] = val
    
    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True 
    
    def freeze_decoder(self):
        for param in self.model.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        for param in self.model.decoder.parameters():
            param.requires_grad = True