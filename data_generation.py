# Import Data Science Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# System libraries
from pathlib import Path
import os.path


train_dir = Path('./data/train')
test_dir = Path('./data/test')
val_dir = Path('./data/valid')

def prepare_data(random_state=42):
    # Get filepaths and labels for training data
    train_filepaths = list(train_dir.glob(r'**/*.JPG')) + list(train_dir.glob(r'**/*.jpg')) + list(train_dir.glob(r'**/*.png')) + list(train_dir.glob(r'**/*.png'))
    train_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], train_filepaths))

    train_filepaths = pd.Series(train_filepaths, name='Filepath').astype(str)
    train_labels = pd.Series(train_labels, name='Label')

    # Concatenate filepaths and labels for training data
    train_df = pd.concat([train_filepaths, train_labels], axis=1)

    # Get filepaths and labels for test data
    test_filepaths = list(test_dir.glob(r'**/*.JPG')) + list(test_dir.glob(r'**/*.jpg')) + list(test_dir.glob(r'**/*.png')) + list(test_dir.glob(r'**/*.png'))
    val_filepaths = list(val_dir.glob(r'**/*.JPG')) + list(val_dir.glob(r'**/*.jpg')) + list(val_dir.glob(r'**/*.png')) + list(val_dir.glob(r'**/*.png'))
    # We combine the given test and validation folders to get a test set
    test_filepaths = test_filepaths + val_filepaths
    test_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], test_filepaths))

    test_filepaths = pd.Series(test_filepaths, name='Filepath').astype(str)
    test_labels = pd.Series(test_labels, name='Label')

    # Concatenate filepaths and labels for training data
    test_df = pd.concat([test_filepaths, test_labels], axis=1)

    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=random_state, stratify=train_df['Label'])
    print('Train Shape: ', train_df.shape)
    print('Validation Shape: ', val_df.shape)
    print('Test Shape: ', test_df.shape)
    return train_df, val_df, test_df

def prepare_from_combined(random_state=42):
    # Get filepaths and labels for training, val, test data
    train_filepaths = list(train_dir.glob(r'**/*.JPG')) + list(train_dir.glob(r'**/*.jpg')) + list(train_dir.glob(r'**/*.png')) + list(train_dir.glob(r'**/*.png'))
    test_filepaths = list(test_dir.glob(r'**/*.JPG')) + list(test_dir.glob(r'**/*.jpg')) + list(test_dir.glob(r'**/*.png')) + list(test_dir.glob(r'**/*.png'))
    val_filepaths = list(val_dir.glob(r'**/*.JPG')) + list(val_dir.glob(r'**/*.jpg')) + list(val_dir.glob(r'**/*.png')) + list(val_dir.glob(r'**/*.png'))

    filepaths = train_filepaths + test_filepaths + val_filepaths
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    df_data = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(df_data, test_size=0.15, shuffle=True, random_state=random_state, stratify=df_data['Label'])
    train_df, val_df = train_test_split(train_df, test_size=0.20, shuffle=True, random_state=random_state, stratify=train_df['Label'])
    
    print('Train Shape: ', train_df.shape)
    print('Validation Shape: ', val_df.shape)
    print('Test Shape: ', test_df.shape)
    return train_df, val_df, test_df