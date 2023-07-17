from Data.Preprocessing import TokenizerSingleton
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import wordnet
import numpy as np
import json

nltk.download("wordnet")

# Concepts for each emotion using adjectives: Simplest case of concepts
# These are enhanced unsing synonyms from nltk to increase numbers
CONCEPTS = {
  'Disgust': ['contempt', 'disgust', 'revulsion'],
  'Envy': ['Envy', 'jealousy'],
  'Exasperation': ['Exasperation', 'frustration'],
  'Irritation': ['Aggravation', 'agitation', 'annoyance', 'grouchiness', 'grumpiness', 'irritation'],
  'Rage': ['Anger', 'bitterness', 'dislike', 'ferocity', 'fury', 'hate', 'hostility', 'loathing', 'outrage', 'rage', 'resentment', 'scorn', 'spite', 'vengefulness', 'wrath'],
  'Torment': ['Torment'],
  'Horror': ['Alarm', 'fear', 'fright', 'horror', 'hysteria', 'mortification', 'panic', 'shock', 'terror'],
  'Nervousness': ['Anxiety', 'apprehension', 'distress', 'dread', 'nervousness', 'tenseness', 'uneasiness', 'worry'],
  'Cheerfulness': ['Amusement', 'bliss', 'cheerfulness', 'delight', 'ecstasy', 'elation', 'enjoyment', 'euphoria', 'gaiety', 'gladness', 'glee', 'happiness', 'jolliness', 'joviality', 'joy', 'jubilation', 'satisfaction'],
  'Contentment': ['Contentment', 'pleasure'],
  'Enthrallment': ['Enthrallment', 'rapture'],
  'Optimism': ['Eagerness', 'hope', 'optimism'],
  'Pride': ['Pride', 'triumph'],
  'Relief': ['Relief'],
  'Zest': ['Enthusiasm', 'excitement', 'exhilaration', 'thrill', 'zeal', 'zest'],
  'Affection': ['Adoration', 'affection', 'attraction', 'caring', 'compassion', 'fondness', 'liking', 'love', 'sentimentality', 'tenderness'],
  'Longing': ['Longing'],
  'Lust': ['Arousal', 'desire', 'infatuation', 'lust', 'passion'],
  'Disappointment':['Disappointment', 'dismay', 'displeasure'],
  'Neglect': ['Alienation', 'defeat', 'dejection', 'embarrassment', 'homesickness', 'humiliation', 'insecurity', 'isolation', 'insult', 'loneliness', 'neglect', 'rejection'],
  'Sadness': ['Depression', 'despair', 'gloom', 'glumness', 'grief', 'hopelessness', 'melancholy', 'misery', 'sadness', 'sorrow', 'unhappiness', 'woe'],
  'Shame': ['Guilt', 'regret', 'remorse', 'shame'],
  'Suffering': ['Agony', 'anguish', 'hurt', 'suffering'],
  'Sympathy': ['Pity', 'sympathy'],
  'Surprise': ['Amazement', 'astonishment', 'surprise'],
}

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
  def __init__(self, texts, tokenizer, max_len):
    self.texts = texts
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    encoding = self.tokenizer.encode_plus(
      self.texts[item],
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      return_attention_mask=True,
      return_tensors='pt',
      padding='max_length',
      truncation=True,
    )

    return torch.stack((encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)))

r'''
This method connstructs concepts from simple adjecvtives which describe emotions and feelings
The pro of using these concepts is that they are simpler and can be enhanced on the go
with more samples. Plus, concepts here are completely new to the model and were not used for training.
'''
def assembleConceptsFromAdjectives(emotions, concepts):
  captumConcepts = []
  tokenizer = TokenizerSingleton.getTokenizerInstance()
  for id, emo in enumerate(emotions):
    conceptSamplesForEmotion = concepts[emo]
    conceptDataset = EmotionConceptContainer(conceptSamplesForEmotion, tokenizer, 128)
    conceptDataLoader = dataset_to_dataloader(conceptDataset)

    captumConcepts.append(Concept(id=id, name=emo, data_iter=conceptDataLoader))

  return captumConcepts

r'''
This method constructs concpets from full sentences from the dataset
It include sentences that were used when training the model.

It randomly samples n sentences for each fine grained label and uses those
as examples for each concept. Concpets, then are the fine grained emotions
'''
def assembleConceptsFromDataset(dataset, nSamplesPerConcept=4):
  emotions = dataset.fineEmo.unique().tolist()
  captumConcepts = []
  tokenizer = TokenizerSingleton.getTokenizerInstance()

  for id, emo in enumerate(emotions):
    samples = dataset[dataset.fineEmo == emo].sentence.to_numpy()
    conceptSamplesForEmotion = np.random.choice(samples, nSamplesPerConcept)

    conceptDataset = EmotionConceptContainer(conceptSamplesForEmotion, tokenizer, 128)
    conceptDataLoader = dataset_to_dataloader(conceptDataset)

    captumConcepts.append(Concept(id=id, name=emo, data_iter=conceptDataLoader))

  return captumConcepts

r'''
Load concepts from a pre-saved JSON file
Expects JSON to include a hash with keys being concept names
and values being arrays of concept samples
'''
def loadConceptsFromFile(filePath):
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