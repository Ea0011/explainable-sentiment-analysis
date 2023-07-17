from Common.Emotions import BROAD_EMOTIONS, FINE_EMOTIONS
from Utils.TextProcessing import TokenizerSingleton, encodeText
from Model.EmoClassifier import TextClassifierModule
from Utils.SingletonModelLoader import SingletonModelLoader
import torch
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients


class AttributionExplainer:
  """
  AttributionExplainer: Class that wraps Captum's Layer Integrated Gradients and
  Layer Conductance attribution methods to work with our pre-traibed classifier

  Attributes
  ----------
  
  Methods
  ----------

  __init__(modelPath):
    Initializes the Xplainer. Expects a path to a checkpoint file for fine classifier

  constructInputReferencePair(text):
    Constructs inputs to Layer Integrated Gradients. Prepares inputs for
    input text and an empty reference that is used to attribute word importance
    while interpolating from empty reference to users input.

  predict(input, attention_mask):
    Computes emotion scores for the given input and attention_mask

  predict_with_embeddings(input_embeds, attention_mask):
    Computes emotion scores for the given input and attention_mask
    but uses input embeddings from bert directly instead of input token ids. Used for Layer Conductance

  summarize_attributions(attributions, all_tokens, exlcude_tokens=[]):
    Given attributions computed for each token, this function aggregates importance
    of each token from all attribution steps (ex. Layer integrated gradients are computed with multiple steps)
    Excludes importance for tokens in exclude_tokens array. This is useful to exclude padding tokens

  interpret_emotion_prediction_top_k(text, top_k=3, n_steps=10):
    Does n_steps of Layer Integrated Gradients with respect to input words for emotions with top_k
    predicted scores.
    Returns the predicted emotion, input, attributions per token and vizualization record for Captum
    Vizualizer.

  construct_whole_bert_embeddings(input_ids, ref_input_ids):
    Constructs BERT embeddings for inputs and reference. Used for Layer Conductance.

  interpret_layer_attributions(text):
    Runs Layer Conductance for given input text. Returns predicted emotion, 
    attributions of tokens of each word through all 12 BERT layers and the tokens
  """
  def __init__(self, modelPath) -> None:
    self.tokenizer = TokenizerSingleton.getTokenizerInstance()
    self.model = SingletonModelLoader.getModelInstance('fine_classifier', modelPath, TextClassifierModule)

  def constructInputReferencePair(self, text):
    encoding = encodeText(text)
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']

    # construct reference token ids 
    ref_input_ids = [self.tokenizer.cls_token_id] + [self.tokenizer.pad_token_id] * (len(input_ids[0]) - 1) # - 1 to exclude already present cls token

    return input_ids, torch.tensor([ref_input_ids]), attention_mask
  
  def summarize_attributions(self, attributions, all_tokens, exlcude_tokens=[]):
    include_indices = [i for i in range(len(all_tokens)) if all_tokens[i] not in exlcude_tokens]

    attributions = attributions[:, include_indices].sum(dim=-1).squeeze(0) # get attributions per token
    attributions = attributions / torch.norm(attributions)
    return attributions

  def predict(self, input, attention_mask):
    output = self.model(input, attention_mask=attention_mask)
    return torch.softmax(output, dim=1)

  def predict_with_embeddings(self, input_embeds, attention_mask):
    prediction = self.model(None, attention_mask, input_embeds=input_embeds)
    return prediction.softmax(dim=1)

  def interpret_emotion_prediction_top_k(self, text, k=3, n_steps=10):
    input_ids, ref_input_ids, attention_mask = self.constructInputReferencePair(text)

    exclude_tokens = [self.tokenizer.pad_token]
    indices = input_ids[0].detach().tolist()
    all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
    tokens = [t for t in all_tokens if t not in exclude_tokens]

    emotion_scores = self.predict(input_ids, attention_mask=attention_mask)
    
    # get top k emotions
    top_preds = torch.topk(emotion_scores, k)
    top_k_emotions = top_preds.indices[0]
    top_k_scores = top_preds.values[0]

    # Predicted emotion
    predicted_emotion = emotion_scores.argmax()

    viz_records = []
    word_attributions_for_emotion = dict()
    for emotion, score in zip(top_k_emotions, top_k_scores):
      # feature attributions for embeddings, each embedding corresponds to one word in the input sequence
      lig = LayerIntegratedGradients(self.predict, self.model.feature_extractor.embeddings.word_embeddings)
      attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(attention_mask),
                                        return_convergence_delta=True,
                                        target=emotion,
                                        n_steps=n_steps)
      
      total_attributions = self.summarize_attributions(attributions, all_tokens, exclude_tokens)
      word_attributions_for_emotion[emotion] = total_attributions

      viz_records.append(viz.VisualizationDataRecord(
                            total_attributions,
                            score, # predicted class prob
                            FINE_EMOTIONS[emotion], # predicted class
                            FINE_EMOTIONS[predicted_emotion.item()], # true class
                            FINE_EMOTIONS[emotion],
                            total_attributions.sum(),       
                            tokens, # all tokens for input
                            delta))

    return {
      'text': text,
      'emotion': FINE_EMOTIONS[predicted_emotion.item()],
      'attributions': word_attributions_for_emotion,
      'vizualization': viz_records,
    }

  def construct_whole_bert_embeddings(self, input_ids, ref_input_ids):
    input_embeddings = self.model.feature_extractor.embeddings(input_ids)
    ref_input_embeddings = self.model.feature_extractor.embeddings(ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

  def interpret_layer_attributions(self, text):
    layer_attrs = []
    input_ids, ref_input_ids, attention_mask = self.constructInputReferencePair(text)

    indices = input_ids[0].detach().tolist()
    all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
    exclude_tokens = [self.tokenizer.pad_token]
    tokens = [t for t in all_tokens if t not in exclude_tokens]

    # # takes long time with a CPU, approx 10 mins
    input_embeddings, ref_input_embeddings = self.construct_whole_bert_embeddings(input_ids, ref_input_ids)
    predicted_emotion = self.predict_with_embeddings(input_embeddings, attention_mask).argmax()

    for i in range(self.model.feature_extractor.config.num_hidden_layers):
      lc = LayerConductance(self.predict_with_embeddings, self.model.feature_extractor.encoder.layer[i])
      layer_attributions = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(attention_mask), target=predicted_emotion)
      layer_attrs.append(self.summarize_attributions(layer_attributions, all_tokens, exclude_tokens).detach().tolist())

    return {
      'layer_attributions': layer_attrs,
      'emotion': FINE_EMOTIONS[predicted_emotion.item()],
      'tokens': tokens,
    }
