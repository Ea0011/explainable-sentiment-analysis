from Utils.TextProcessing import encodeText
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import wordnet
import numpy as np
import json

nltk.download("wordnet")

# Obtaining concept candidates for tCAV
def getSynonymsAndExamples(word):
  wordDescription = wordnet.synsets(word)
  synonyms = []
  examples = []

  for w in wordDescription:
    lemmas = w.lemmas()
    for l in lemmas:
      synonym = l.name()
      synonym = ' '.join(synonym.split('_'))
      synonyms.append(synonym)
    samples = w.examples()
    examples.extend(samples)

  return np.unique(synonyms).tolist(), np.unique(examples).tolist()

class EmotionConceptContainer(Dataset):
  """
  EmotionConceptContainer: A class that encapuslates concepts for emotions
  Methods
  ----------

  __getitem__():
    Retrieve and preporcess the conversation

  __len__():
    Gets the length of the whole dataset
  """
  def __init__(self, texts, max_len):
    self.texts = texts
    self.max_len = max_len
  
  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    encoding = encodeText(item)
    return torch.stack((encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)))

def assembleConceptsFromAdjectives(emotions, concepts):
  r'''
  This method connstructs concepts from simple adjecvtives which describe emotions and feelings
  The pro of using these concepts is that they are simpler and can be enhanced on the go
  with more samples. Plus, concepts here are completely new to the model and were not used for training.
  '''
  captumConcepts = []
  for id, emo in enumerate(emotions):
    conceptSamplesForEmotion = concepts[emo]
    conceptDataset = EmotionConceptContainer(conceptSamplesForEmotion, 128)
    conceptDataLoader = dataset_to_dataloader(conceptDataset)

    captumConcepts.append(Concept(id=id, name=emo, data_iter=conceptDataLoader))

  return captumConcepts


def assembleConceptsFromDataset(dataset, nSamplesPerConcept=4):
  r'''
  This method constructs concpets from full sentences from the dataset
  It include sentences that were used when training the model.

  It randomly samples n sentences for each fine grained label and uses those
  as examples for each concept. Concpets, then are the fine grained emotions
  '''
  emotions = dataset.fineEmo.unique().tolist()
  captumConcepts = []

  for id, emo in enumerate(emotions):
    samples = dataset[dataset.fineEmo == emo].sentence.to_numpy()
    conceptSamplesForEmotion = np.random.choice(samples, nSamplesPerConcept)

    conceptDataset = EmotionConceptContainer(conceptSamplesForEmotion, 128)
    conceptDataLoader = dataset_to_dataloader(conceptDataset)

    captumConcepts.append(Concept(id=id, name=emo, data_iter=conceptDataLoader))

  return captumConcepts


def loadConceptsFromFile(filePath):
  r'''
  Load concepts from a pre-saved JSON file
  Expects JSON to include a hash with keys being concept names
  and values being arrays of concept samples
  '''
  with open(filePath, 'r') as conceptFile:
    return json.load(conceptFile)

# Enhance concepts with some more synonyms
def enhanceConceptsWithSynonyms(concepts):
  newConcepts = concepts.copy()

  for c, words in newConcepts.items():
    synonyms = []
    for w in words:
      syns, _ = getSynonymsAndExamples(w)
      synonyms.extend(syns)
    
    newConcepts[c].extend(synonyms)
    newConcepts[c] = np.unique(newConcepts[c]).tolist()

  return newConcepts