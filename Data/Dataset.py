from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder




class SentenceEmotionDataset(Dataset):
  """
  EmotionContextDataset: A class that encapsulates chat data
  The class reads the conversation, normalizes it for BERTTweet input,
  puts in classification token.

  Attributes
  ----------
  
  Methods
  ----------

  __getitem__():
    Retrieve and preporcess the conversation

  __len__():
    Gets the length of the whole dataset
  """
  def __init__(self, texts, labels, tokenizer, max_len):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.sentences = texts
  
  def __len__(self):
    return len(self.sentences)
  
  def __getitem__(self, item):
    text = str(self.sentences[item])
    label = self.labels[item]

    
    #Puts CLS token at the beginning of the sentece
    #Puts eos token at the end of the sentence
    #pad with 0s until the max length
    #Also returns corresponding mask!
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      return_attention_mask=True,
      return_tensors='pt',
      padding='max_length',
      truncation=True,
    )

    return {
      'content_text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(label, dtype=torch.long)
    }




class TextDataModule(pl.LightningDataModule):
  """
  TextDataModule: Takes in train/val/test datasets and prepares dataloaders for training
  with pytorch lightning. An instance of data module is necessary to train with pytorch
  lightning.

  Attributes
  ----------
  train_data/val_data/test_data : Dataset
    Containers for Datasets
  

  Methods
  ----------

  train/val/test_loader():
    Return appropriate data loaders for each data set partition
  """

  def __init__(self, train_set, val_set, test_set, batch_size: 32, **kwargs) -> None:
    super().__init__()
    self.train_data = train_set
    self.val_data = val_set
    self.test_data = test_set
    self.batch_size = batch_size

  def setup(self, stage = None):
    self.training_set = self.train_data
    self.validation_set = self.val_data
    self.test_set = self.test_data

  def train_dataloader(self):
    return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

  def val_dataloader(self):
    return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

  def test_dataloader(self):
    return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)




def create_data_module(df_train, df_val, df_test, tokenizer, max_len, batch_size):
  """
  Creates pytorch lightning data module for training the model
  This function takes in split train/val/test datasets and prepares a data
  module.

  The function constructs Dataset objects out of input raw data,
  encodes the labels and creates a data module with all dataloaders.
  """
  labels_train = df_train.broadEmo.to_numpy()
  labels_val = df_val.broadEmo.to_numpy()
  labels_test = df_test.broadEmo.to_numpy()
  
  integer_encoded_train = LabelEncoder().fit_transform(labels_train)
  integer_encoded_val = LabelEncoder().fit_transform(labels_val)
  integer_encoded_test = LabelEncoder().fit_transform(labels_test)

  train_set = SentenceEmotionDataset(
    texts=df_train.sentence.to_numpy(),
    labels=integer_encoded_train,
    tokenizer=tokenizer,
    max_len=max_len
  )

  val_set = SentenceEmotionDataset(
    texts=df_val.sentence.to_numpy(),
    labels=integer_encoded_val,
    tokenizer=tokenizer,
    max_len=max_len
  )

  test_set = SentenceEmotionDataset(
    texts=df_test.sentence.to_numpy(),
    labels=integer_encoded_test,
    tokenizer=tokenizer,
    max_len=max_len
  )

  return TextDataModule(train_set, val_set, test_set, batch_size)

def constructModelInput(text)