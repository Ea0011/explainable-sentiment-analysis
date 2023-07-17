r'''
Concepts for each emotion using adjectives: Simplest case of concepts
These are enhanced unsing synonyms from nltk to increase numbers
'''
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