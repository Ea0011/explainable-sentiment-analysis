
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Data.Dataset import SentenceEmotionDataset, TextDataModule
from sklearn.preprocessing import LabelEncoder


def create_data_module(df_train, df_val, df_test, label_name, batch_size):
  """
  Creates pytorch lightning data module for training the model
  This function takes in split train/val/test datasets and prepares a data
  module.

  The function constructs Dataset objects out of input raw data,
  encodes the labels and creates a data module with all dataloaders.
  """
  labels_train = df_train.loc[:,label_name].to_numpy()
  labels_val = df_val.loc[:,label_name].to_numpy()
  labels_test = df_test.loc[:,label_name].to_numpy()
  
  integer_encoded_train = LabelEncoder().fit_transform(labels_train)
  integer_encoded_val = LabelEncoder().fit_transform(labels_val)
  integer_encoded_test = LabelEncoder().fit_transform(labels_test)

  train_set = SentenceEmotionDataset(
    texts=df_train.sentence.to_numpy(),
    labels=integer_encoded_train,
  )

  val_set = SentenceEmotionDataset(
    texts=df_val.sentence.to_numpy(),
    labels=integer_encoded_val,
  )

  test_set = SentenceEmotionDataset(
    texts=df_test.sentence.to_numpy(),
    labels=integer_encoded_test,
  )

  return TextDataModule(train_set, val_set, test_set, batch_size)

def prepare_data(
    data_path,
    label_name,
    dataset_names,
    split_train_val_test,
    batch_size,
    RANDOM_SEED = 5,
):

    data_df = pd.read_csv(data_path, sep='\t')

    #Get classes
    try:
        classes = data_df[label_name].unique().tolist()
    except:
        print("You entered a not valid label category, please select either broadEmo or fineEmo")
        return

    #Get datasets
    if all([(elem in np.unique(data_df["dataset"])) for elem in dataset_names]):
        Data = data_df[data_df.dataset.isin(np.unique(dataset_names))]
    else:
        print("You entered a non existing dataset name, please select one from",np.unique(data_df["dataset"]))

    #TODO: Bertweet normalization?!
    #https://huggingface.co/docs/transformers/master/en/model_doc/bertweet

    #TODO: Either improve or delete this part
    """
    #Find the max length here:
    #--->if max_length_check:
    #--->Do this inplace?
    #--->it is putting cls token and eos tokens by itself, right?
    token_lens = []
    for txt in Data.sentence:
        tokens = tokenizer.encode(txt)
        token_lens.append(len(tokens))
    #--->print max length    
    print("max length:",max(token_lens))
    #--->print how many samples are longer than the data threshold (out of how many examples)
    arr = np.array(token_lens)
    print("how many samples exceeds theshold:",sum(arr[arr > max_len]),"out of",len(token_lens))
    #--->What to do with them? (truncate)
    """

    #Do the split
    df_train,df_val = train_test_split(Data, train_size=split_train_val_test[0], random_state=RANDOM_SEED)
    df_val,df_test = train_test_split(df_val, train_size = split_train_val_test[1]/(split_train_val_test[1]+split_train_val_test[2]), random_state=RANDOM_SEED)
    print("Training data shape: ",df_train.shape, "\nValidation data shape: ",df_val.shape, "\nTest data shape: ",df_test.shape)

    #TODO: Save the split (---> also implement load)
    #--->Maybe add a new column for train/val/test label
    #--->Also new column for encoded parts

    #TODO: Improve create_data_module:
    #--->A bit redundent
    #--->Do some of the stuff here, especially like encoder.plus, then maybe save it
    #--->Some stuff are hardcoded, fix this
    #--->Send the LabelEncoder from here or do that compeletly here (like inplace or new cloumn)
    #--->Save the LabelEncoder's mapping
    #--->Batchsize 32 for validation and test sets, is it better or worse
    data_module = create_data_module(df_train, df_val, df_test, label_name, batch_size)


    return data_module,classes #,Data etc.
