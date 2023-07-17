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
