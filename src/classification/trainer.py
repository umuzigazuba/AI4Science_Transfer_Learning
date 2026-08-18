import torch
import lightning

from sklearn.metrics import f1_score


class MultiClassClassifier(lightning.LightningModule):

    def __init__(self, model: torch.nn.Module, optimiser: torch.optim, learning_rate: float, 
                 sheduler_name: str, max_epochs: int, weight = torch.Tensor):

        super().__init__()

        self.model = model

        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.sheduler_name = sheduler_name
        self.max_epochs = max_epochs

        self.weight = weight

        self.validation_predictions = []
        self.validation_targets = []


    def forward(self, input: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:

        return self.model(input, meta)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        optim = self.optimiser(self.parameters(), lr = self.learning_rate, weight_decay = 1e-4)
        
        if self.sheduler_name == "CosineAnnealingLR": 
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = self.max_epochs)
            return [optim], [lr_scheduler]

        elif self.sheduler_name == "ReduceLROnPlateau": 
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode = "min", factor = 0.1, patience = 5)
            return [optim], [lr_scheduler]

        else: 
            return optim
    
    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:

        return torch.nn.functional.cross_entropy(
            predictions.float(), targets.float(), weight = self.weight.to(self.device)
        )

    def compute_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lc_data, meta, targets = batch
        predictions = self(lc_data, meta)
        loss = self.compute_loss(predictions, targets)

        return loss, predictions, targets

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:

        loss, _, _ = self.compute_step(batch)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss}  # necessary for training, needs to be called loss!

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):

        loss, predictions, targets = self.compute_step(batch)
        self.log(
            "validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.validation_predictions.append(torch.argmax(predictions, dim = 1))
        self.validation_targets.append(targets)

    def on_validation_epoch_end(self):

        predictions = torch.cat(self.validation_predictions)
        targets = torch.cat(self.validation_targets)

        validation_accuracy = (predictions == targets).float().mean().item()
        validation_f1_score = f1_score(targets.float().cpu().numpy(), predictions.float().cpu().numpy(), average = None)[-1]
        
        self.log_dict(
            {
                "validation_accuracy": validation_accuracy,
                "validation_f1_score": validation_f1_score,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        self.validation_predictions.clear()
        self.validation_targets.clear()

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        lc_data, meta, _ = batch

        logits = self(lc_data, meta)
        predictions = torch.argmax(logits, dim = 1)

        return predictions
