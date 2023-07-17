from transformers import AutoTokenizer


# Max sentence length which is understood by BERTWeet
MAX_TEXT_LEN = 128

class TokenizerSingleton:
  r'''
  A singleton class which provides access to the tokenizer from BERTweet
  Tokenizer can be accessed globally and it is not reloaded each time it is
  being accessed
  '''
  tokenizer = None
  @staticmethod
  def getTokenizerInstance():
    if TokenizerSingleton.tokenizer == None:
      TokenizerSingleton()
    return TokenizerSingleton.tokenizer
  
  def __init__(self):
    """ Virtually private constructor. """
    if TokenizerSingleton.tokenizer != None:
      raise Exception("This class is a singleton!")
    else:
      TokenizerSingleton.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)


def encodeText(text):
  r'''
  Encodes the text for input to BERTweet transformer.
  Tokenizer from pre-trained BERTweet is used for encoding the sentence
  The sentence is tokenized with vocabulary used by BERTweet
  Truncation, padding to max length and handling of special tokens are done by
  the tokenizer from BERTweet

  ---------
  Returns a hash:
  encoded['input_ids'] -> sequence with token ids for each word
  encoded['attention_mask'] -> mask which makes the model attend to only words (excludes padding)
  '''
  tokenizer = TokenizerSingleton.getTokenizerInstance()

  encoding = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=MAX_TEXT_LEN,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
  )

  return encoding