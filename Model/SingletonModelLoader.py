r'''
Class that load models from checkpoints and stores them in memory
Stored models can be accessed without having to reload them
globally in the application

They will not be reloaded each time they are accessed

modelKlass -> Class of model to be loaded
model -> unique name to store the model under, used to retriev loaded model
path -> path of checkpoint file of the model
'''
class SingletonModelLoader:
  models = {}

  @staticmethod
  def getModelInstance(model, path=None, modelKlass=None):
    if model not in SingletonModelLoader.models:
      SingletonModelLoader(model, path, modelKlass)

    return SingletonModelLoader.models[model]
  
  def __init__(self, model, path, modelKlass):
    """ Virtually private constructor. """
    if model in SingletonModelLoader.models:
      raise Exception("This class is a singleton!")
    else:
      SingletonModelLoader.models[model] = modelKlass.load_from_checkpoint(path, map_location='cpu')
      SingletonModelLoader.models[model].eval()