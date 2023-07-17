from Utils.TextProcessing import encode_text
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
def get_synonyms_and_examples(word):
  word_description = wordnet.synsets(word)
  synonyms = []
  examples = []

  for w in word_description:
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
    encoding = encode_text(item)
    return torch.stack((encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)))

def assemble_concepts_from_adjectives(emotions, concepts):
  r'''
  This method connstructs concepts from simple adjecvtives which describe emotions and feelings
  The pro of using these concepts is that they are simpler and can be enhanced on the go
  with more samples. Plus, concepts here are completely new to the model and were not used for training.
  '''
  captum_concepts = []
  for id, emo in enumerate(emotions):
    concept_samples_for_emotion = concepts[emo]
    concept_dataset = EmotionConceptContainer(concept_samples_for_emotion, 128)
    concept_data_loader = dataset_to_dataloader(concept_dataset)

    captum_concepts.append(Concept(id=id, name=emo, data_iter=concept_data_loader))

  return captum_concepts


def assemble_concepts_from_dataset(dataset, n_samples_per_concept=4):
  r'''
  This method constructs concpets from full sentences from the dataset
  It include sentences that were used when training the model.

  It randomly samples n sentences for each fine grained label and uses those
  as examples for each concept. Concpets, then are the fine grained emotions
  '''
  emotions = dataset.fineEmo.unique().tolist()
  captum_concepts = []

  for id, emo in enumerate(emotions):
    samples = dataset[dataset.fineEmo == emo].sentence.to_numpy()
    concept_samples_for_emotion = np.random.choice(samples, n_samples_per_concept)

    concept_dataset = EmotionConceptContainer(concept_samples_for_emotion, 128)
    concept_data_loader = dataset_to_dataloader(concept_dataset)

    captum_concepts.append(Concept(id=id, name=emo, data_iter=concept_data_loader))

  return captum_concepts


def load_concepts_from_file(file_path):
  r'''
  Load concepts from a pre-saved JSON file
  Expects JSON to include a hash with keys being concept names
  and values being arrays of concept samples
  '''
  with open(file_path, 'r') as concept_file:
    return json.load(concept_file)

# Enhance concepts with some more synonyms
def enhance_concepts_with_synonyms(concepts):
  new_concepts = concepts.copy()

  for c, words in new_concepts.items():
    synonyms = []
    for w in words:
      syns, _ = get_synonyms_and_examples(w)
      synonyms.extend(syns)
    
    new_concepts[c].extend(synonyms)
    new_concepts[c] = np.unique(new_concepts[c]).tolist()

  return new_concepts