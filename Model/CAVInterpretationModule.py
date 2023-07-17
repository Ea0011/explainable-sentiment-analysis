from Model.EmoClassifier import TextClassifierModule

r'''
Overrides the forward function to accept single tensor of inputs
This is neccessary to be able to use Captum API-s

inputs[:, 0] -> input_token_ids
inputs[:, 1] -> attention_mask
'''
class CAVInterpretationModule(TextClassifierModule):
  def forward(self, inputs):
    attention_mask = inputs[:, 1]
    token_ids = inputs[:, 0]

    x = self.feature_extractor(input_ids=token_ids, attention_mask=attention_mask).pooler_output

    for _, layer in enumerate(self.classifier):
      x = layer(x)

    return x