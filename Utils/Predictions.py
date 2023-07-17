from Utils.TextProcessing import encode_text
import torch

def predict_emotion_with_probs(model, labels, text):
  encoding = encode_text(text)
  input_ids = encoding['input_ids'].flatten().unsqueeze(0)
  attention_mask = encoding['attention_mask'].flatten().unsqueeze(0)

  emotion_scores = model(input_ids, attention_mask)
  emotion = torch.argmax(emotion_scores)

  class_probabilities = {}
  with torch.no_grad():
    probs = torch.softmax(emotion_scores, dim=1).flatten()
    for prob, label in zip(probs, labels):
      class_probabilities[label] = "{:.2%}".format(prob.item())

  return labels[emotion], class_probabilities

def compute_model_predictions(model, data_loader):
  ground_truth = torch.tensor([], dtype=torch.long)
  predictions = torch.tensor([], dtype=torch.long)

  for x in data_loader:
    input_ids, attention_mask, targets = x['input_ids'], x['attention_mask'], x['targets']

    emotion_scores = model(input_ids, attention_mask)
    predictions = torch.cat((predictions, emotion_scores.argmax(dim=1)))
    ground_truth = torch.cat((ground_truth, targets))

  return predictions, ground_truth
