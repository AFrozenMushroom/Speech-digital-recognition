import os
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pyaudio
import wave
import pygame
from tensorflow.keras import layers
from tensorflow.keras import models
import IPython.display as ipd

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'Audio'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'audio',
      origin="C:\PYTHONDATA\Speech digital recognition\Audio",
      extract=False,
      cache_dir='.', cache_subdir='data')

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')


label_names = np.array(train_ds.class_names)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

example_audio=[]
example_labels=[]
for i,j in test_ds.take(1):
    example_audio.append(i)
    example_labels.append(j)

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

example_spectrograms=[]
example_spect_labels=[]
for i, j in test_spectrogram_ds.take(1):
    example_spectrograms.append(i)
    example_spect_labels.append(j)

model = tf.keras.models.load_model('my_model.keras')

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(example_spectrograms)

i = random.randint(0,63)
a=np.argmax(predictions[i])
print('模型预测结果',a)
print('实际结果',example_spect_labels[0][i])

ipd.Audio(example_audio[0][i].numpy(), rate=16000)