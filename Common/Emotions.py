import numpy as np

r'''
Sorted fine and coarse grained emotions from Empathetic Dialogues
Can be used to translate prediction index back emotion label
'''
BROAD_EMOTIONS = np.array(['Anger', 'Fear', 'Joy', 'Love', 'Other', 'Sad', 'Surprise'])

FINE_EMOTIONS = np.array(['anger', 'annoyance', 'anticipating', 'apprehensive', 'ashamed',
  'caring', 'confident', 'content', 'devastated', 'disappointment',
  'disgusted', 'embarrassment', 'excitement', 'faithful', 'fear',
  'furious', 'gratitude', 'guilty', 'hopeful', 'impressed',
  'jealous', 'joy', 'lonely', 'nostalgic', 'prepared', 'pride',
  'sadness', 'sentimental', 'surprise', 'terrified', 'trusting',
  'worry'
])

r'''
The mapping used to translate fine grained emotions to coarse grained ones.
The mapping is based on Parrot's emotion classification hierarchy.
'''
FINE_TO_BROAD_MAPPING = {
  "sentimental": "Love",
  "afraid": "Fear",
  "proud": "Joy",
  "faithful": "Joy",
  "terrified": "Fear",
  "joyful": "Joy",
  "angry": "Anger",
  "sad": "Sad",
  "jealous": "Anger",
  "grateful": "Joy",
  "prepared": "Other",
  "embarrassed": "Sad",
  "excited": "Joy",
  "annoyed": "Anger",
  "lonely": "Sad",
  "ashamed": "Sad",
  "guilty": "Sad",
  "surprised": "Surprise",
  "nostalgic": "Joy",
  "confident": "Joy",
  "furious": "Anger",
  "disappointed": "Sad",
  "caring": "Love",
  "trusting": "Other",
  "disgusted": "Anger",
  "anticipating": "Other",
  "anxious": "Fear",
  "hopeful": "Joy",
  "content": "Joy",
  "impressed": "Surprise",
  "apprehensive": "Fear",
  "devastated": "Sad"
}

r'''
For the sake of coherence of fine grained emotions while combining several datasets together,
The mapping below was used to coerce similar emotions from different datasets.

For example: 
happiness -> joy
disappointed -> dissapointment
etc.

The mapping is done in a way to preserve original labels but be able to combine
several datasets together without colliding labels such as dissapointed and dissapointment.
'''
FINE_EMOTIONS_TRANSLATION = {
  "neutral": "neutral",
  "anger": "anger",
  "fear": "fear",
  "annoyance": "annoyance",
  "surprise": "surprise",
  "gratitude": "gratitude",
  "desire": "desire",
  "admiration": "admiration",
  "confusion": "confusion",
  "amusement": "joy",
  "caring": "caring",
  "embarrassment": "embarrassment",
  "grief": "grief",
  "joy": "joy",
  "sadness": "sadness",
  "curiosity": "curiosity",
  "disapproval": "disapproval",
  "optimism": "optimism",
  "approval": "approval",
  "excitement": "excitement",
  "disappointment": "disappointment",
  "remorse": "guilty",
  "love": "love",
  "realization": "realization",
  "disgust": "disgust",
  "relief": "relief",
  "pride": "pride",
  "nervousness": "nervousness",
  "empty": "empty",
  "enthusiasm": "enthusiasm",
  "worry": "worry",
  "fun": "joy",
  "hate": "hate",
  "happiness": "joy",
  "boredom": "boredom",
  "guilty": "guilty",
  "lonely": "lonely",
  "excited": "excitement",
  "sad": "sadness",
  "hopeful": "hopeful",
  "angry": "anger",
  "joyful": "joy",
  "disappointed": "disappointment",
  "faithful": "faithful",
  "content": "content",
  "annoyed": "annoyance",
  "terrified": "terrified",
  "nostalgic": "nostalgic",
  "grateful": "gratitude",
  "trusting": "trusting",
  "surprised": "surprise",
  "ashamed": "ashamed",
  "impressed": "impressed",
  "proud": "pride",
  "furious": "furious",
  "sentimental": "sentimental",
  "confident": "confident",
  "anxious": "worry",
  "jealous": "jealous",
  "afraid": "fear",
  "disgusted": "disgusted",
  "embarrassed": "embarrassment",
  "anticipating": "anticipating",
  "devastated": "devastated",
  "prepared": "prepared",
  "apprehensive": "apprehensive"
}
