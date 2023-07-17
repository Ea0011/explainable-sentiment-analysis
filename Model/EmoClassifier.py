import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import AutoModel



class TextClassifierModule(pl.LightningModule):
  """
  TextClassifierModule: A pytorch lightning module that implement sequence classification

  Attributes
  ----------
  hparams : dict
    Contains neural network hyperparameters. These parameters can be used to
    define the model. Parameters are also saved with model checkpoints to load
    them later
  feature_extractor : RobertaModel
    A roberta model used as a feature extractor
  logger : TensorBoardLogger
    An object that can be used to log training progress to tensorboard.
    Can log scalars as well as histograms, images and computational graphs to
    tensorboard.

  Methods
  -------
  forward(input_ids, attention_mask)
      Forward pass of the network

  general_step(input_ids, attention_mask)
      Computes forward pass for a batch and logs metrics
  general_end(loss)
      Logs metrics after each validation epoch
  training_step(batch, batch_idx)
      Called for each batch during training by Pytorch Lightning
  validation_step(batch, batch_idx)
      Called for each batch during validation
  configure_optimizers()
      Return the optimizer to be used during training. Can optionally configure
      learning rate schedulers
  log(key, value)
      An interface to log values to tensorboard for tracing 
  """

  def __init__(self, n_classes, **hparams):
    super(TextClassifierModule, self).__init__()

    # save network hyperparameters to checkpoints
    self.save_hyperparameters()

    # Used to log compute graph to tensorboard
    self.example_input_array = [
      torch.zeros((1, 128), dtype=torch.long),
      torch.zeros((1, 128), dtype=torch.long),
    ]
    
    self.feature_extractor = AutoModel.from_pretrained("vinai/bertweet-base")
    self.classifier = nn.ModuleList([
      nn.Linear(self.feature_extractor.config.hidden_size, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, n_classes),
    ])

    self.loss_func = F.cross_entropy

    # Freeze feature extractor depending on params
    # Freeze until the layer specified in hparams
    if self.hparams.freeze_feature_extractor:
      for layer in self.feature_extractor.encoder.layer[:self.hparams.freeze_until_layer]:
          for param in layer.parameters():
            param.requires_grad = False

      # freeze all embeddings for now
      for param in self.feature_extractor.embeddings.parameters():
        param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    x = self.feature_extractor(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    for _, layer in enumerate(self.classifier):
      x = layer(x)

    return x

  def general_step(self, batch, batch_idx, mode="train"):
    inputs_ids, targets, attention_mask = batch["input_ids"], batch["targets"], batch['attention_mask']
    
    predicted_label = self.forward(inputs_ids, attention_mask)

    loss = F.cross_entropy(predicted_label, targets)

    correct_items = torch.sum(torch.argmax(predicted_label, dim=1) == targets)
    return loss, correct_items / len(targets)
  
  def general_end(self, loss, mode):
    avg_loss = torch.stack([x for x in loss]).mean()
    return avg_loss

  def training_step(self, batch, batch_idx):
    loss, accuracy = self.general_step(batch, batch_idx)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
    self.log('train_accuracy', accuracy * 100, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

    return loss
  
  def validation_step(self, batch, batch_idx):
    loss, accuracy = self.general_step(batch, batch_idx, mode="val")
    self.log('val_loss', loss, prog_bar=True, batch_size=self.hparams.batch_size)
    self.log('val_accuracy', accuracy * 100, prog_bar=True, batch_size=self.hparams.batch_size)

    return loss

  def validation_epoch_end(self, outputs):
    avg_loss = self.general_end(outputs, "val")
    print("Avg-Loss={}".format(avg_loss))
    tensorboard_logs = {'val/loss': avg_loss}

    return {'val_loss': avg_loss, 'log': tensorboard_logs}

  def configure_optimizers(self):
    optimizer = None

    if self.hparams.optimizer == "SGD":
      optimizer = torch.optim.SGD(
        [
          { "params": self.feature_extractor.parameters(), 'lr': self.hparams.featuer_extactor_lr },
          { "params": self.classifier.parameters() },
        ],
        momentum=0.9,
        lr=self.hparams.learning_rate,
        weight_decay = self.hparams.weight_decay,
      )
    if self.hparams.optimizer == "Adam":
      optimizer = torch.optim.Adam(
        [
          { "params": self.feature_extractor.parameters(), 'lr': self.hparams.featuer_extactor_lr },
          { "params": self.classifier.parameters() },
        ],
        lr=self.hparams.learning_rate,
        weight_decay = self.hparams.weight_decay,
      )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-10, verbose=True, factor=0.1)

    return { "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss" }