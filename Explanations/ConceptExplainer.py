from Data.Preprocessing import TokenizerSingleton
from Model.CAVInterpretationModule import CAVInterpretationModule
from Model.SingletonModelLoader import SingletonModelLoader
import torch
from captum.attr import LayerActivation
from captum.concept._utils.classifier import Classifier, DefaultClassifier
from captum._utils.models.linear_model import model as linear_models
from captum.concept import TCAV
from captum.concept._utils.common import concepts_to_str

class CustomLinearClassifier(DefaultClassifier):
  def __init__(self):
    super().__init__()
    self.lm = linear_models.SkLearnSGDClassifier(loss="hinge", fit_intercept=True, early_stopping=False,
                            learning_rate="optimal", n_jobs=2, class_weight='balanced')

class ConceptExplainer():
  def __init__(self, modelPath, cavsPath, layer, concepts):
    self.concepts = concepts
    self.layer = layer
    self.model = SingletonModelLoader.getModelInstance('broad_classifier', modelPath, CAVInterpretationModule)
    self.cavs = TCAV(model=self.model, layers=[layer], save_path=cavsPath,
                classifier=CustomLinearClassifier())

    _, self.avs = self.cavs.load_cavs(concepts)

  def constructModelInput(self, text):
    tokenizer = TokenizerSingleton.getTokenizerInstance()

    encoding = tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=128,
      return_token_type_ids=False,
      return_attention_mask=True,
      return_tensors='pt',
      padding='max_length',
      truncation=True,
    )

    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']

    return torch.stack((input_ids, attention_mask), dim=1)

  def computeCavs(self):
    self.cavs.compute_cavs([self.concepts], processes=2, force_train=False)
  
  def getCavWeights(self):
    key = concepts_to_str(self.concepts)
    cavWeights = self.cavs.cavs[key][self.layer].stats['weights']
    
    return cavWeights

  def computeActivatedCavs(self, text, target):
    input = self.constructModelInput(text)
    return self.cavs.interpret(input, experimental_sets=[self.concepts], target=target)
  
  def computeCAVSimilarity(self, cavs, text):
    input = self.constructModelInput(text)
    layerActivation = LayerActivation(self.model, self.model.classifier[2])
    activation = layerActivation.attribute(input)

    return torch.cosine_similarity(cavs, activation, dim=-1)

  def getMostSimilarConcepts(self, text, topk):
    concepts = []
    cavWeights = self.getCavWeights()
    similarities = self.computeCAVSimilarity(cavWeights, text)

    for idx in similarities.topk(topk).indices:
      concepts.append(self.concepts[idx])

    return concepts