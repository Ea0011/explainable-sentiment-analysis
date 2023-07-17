from Common.Emotions import BROAD_EMOTIONS
from Utils.TextProcessing import encode_text
from Model.CAVInterpretationModule import CAVInterpretationModule
from Utils.SingletonModelLoader import SingletonModelLoader
import torch
from captum.attr import LayerActivation
from captum.concept._utils.classifier import Classifier, DefaultClassifier
from captum._utils.models.linear_model import model as linear_models
from captum.concept import TCAV
from captum.concept._utils.common import concepts_to_str

class CustomLinearClassifier(DefaultClassifier):
  r'''
  A custom defined linear classifier used to compute concept activation vectors
  '''
  def __init__(self):
    super().__init__()
    self.lm = linear_models.SkLearnSGDClassifier(loss="hinge", fit_intercept=True, early_stopping=False,
                            learning_rate="optimal", n_jobs=2, class_weight='balanced')

class ConceptExplainer():
  r'''
  A class that hosts logic for computing and using CAVs to interpret model predictions.
  Existing CAVs computed using Captum can be loaded from storage. Otherwise, one can
  compute new CAVs through this class.
  --------

  Methods:
    __init__(modelPath, cavsPath, layer, concepts):
      modelPath -> path to the checkpoint file from where torch model should be loaded

      cavsPath -> path to where existing CAVs are saved or where new CAVs should be stored 

      layer -> the hidden layer to use to compute CAVs and explanations  

      concepts -> list of Captum concepts that were used for computing CAVs. Necessary to load existing CAVS. 
      If no previous CAVs exist for given emotions, new ones should be computed  
    
    constructModelInput(self, text):
      constructs model input from a given text
    
    computeCavs(self):
      computes new CAVs for given concepts

    getCavWeights(self):
      gets CAV vectors for given concepts. CAVs should already be computed to use this method

    computeActivatedCavs(self, text, target):
      computes directional derivative in direction of CAVs with respect to the target emotion. If the
      directional derivative correlates with a CAV, it means that the concept influences the decision of the
      model for the target emotion. This means that the concept is relevant and it gets 'activated'.

      This function computes which concepts are 'activated' for the given target emotion prediction.
      Useful to test whether predictions are coherent with the concepts that are 'activated'.

    computeCAVSimilarity(self, cavs, text):
      computes cosine similarity between the activations in the hidden layer for the text and CAVs.
      More similarity means that the concept is prevalent in the text.

      Useful for intepretation purposes to see which kind of concepts decide the output of the model.

    getMostSimilarConcepts(self, text, topk):
      computes top k similar concepts for explaining model decision
  '''
  def __init__(self, modelPath, cavsPath, layer, concepts):
    self.concepts = concepts
    self.layer = layer
    self.model = SingletonModelLoader.get_model_instance('broad_classifier', modelPath, CAVInterpretationModule)
    self.tcav = TCAV(model=self.model, layers=[layer], save_path=cavsPath,
                classifier=CustomLinearClassifier())

    _, self.avs = self.tcav.load_cavs(concepts)

  def construct_model_input(self, text):
    encoding = encode_text(text)
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']

    return torch.stack((input_ids, attention_mask), dim=1)

  def compute_cavs(self):
    self.tcav.compute_cavs([self.concepts], processes=2, force_train=False)
  
  def get_cav_weights(self):
    key = concepts_to_str(self.concepts)
    cav_weights = self.tcav.cavs[key][self.layer].stats['weights']
    
    return cav_weights

  def compute_activated_cavs(self, text, target):
    input = self.construct_model_input(text)
    return self.tcav.interpret(input, experimental_sets=[self.concepts], target=target)

  def predict(self, text):
    input = self.construct_model_input(text)
    prediction = self.model(input)
    emotion = prediction.argmax()

    return BROAD_EMOTIONS[emotion]
  
  def compute_cav_similarity(self, cavs, text):
    input = self.construct_model_input(text)
    layerActivation = LayerActivation(self.model, self.model.classifier[2])
    activation = layerActivation.attribute(input)

    return torch.cosine_similarity(cavs, activation, dim=-1)

  def get_most_similar_concepts(self, text, topk):
    concepts = []
    cav_weights = self.get_cav_weights()
    similarities = self.compute_cav_similarity(cav_weights, text)

    top_concepts = similarities.topk(topk)
    scores, indices = top_concepts.values, top_concepts.indices
    for score, idx in zip(scores, indices):
      concepts.append((self.concepts[idx].name, score))

    return concepts

  def interpret_with_concepts(self, text, top_k=3):
    emotion = self.predict(text)
    concepts = self.get_most_similar_concepts(text, top_k)

    return {
      "emotion": emotion,
      "concepts": concepts,
    }