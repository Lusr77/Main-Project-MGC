from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import datetime


import tensorflow as tf
import os
import librosa
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew


from joblib import load
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
def get_features(y, sr, n_fft = 1024, hop_length = 512):
    # Features to concatenate in the final dictionary
    features = {'centroid': None, 'roloff': None, 'flux': None, 'rmse': None,
                'zcr': None, 'contrast': None, 'bandwidth': None, 'flatness': None}
    
    # Count silence
    if 0 < len(y):
        y_sound, _ = librosa.effects.trim(y, frame_length=n_fft, hop_length=hop_length)
    features['sample_silence'] = len(y) - len(y_sound)

    # Using librosa to calculate the features
    features['centroid'] = librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['roloff'] = librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['rmse'] = librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['flux'] = librosa.onset.onset_strength(y=y, sr=sr).ravel()
    features['contrast'] = librosa.feature.spectral_contrast(y, sr=sr).ravel()
    features['bandwidth'] = librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['flatness'] = librosa.feature.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel()
    
    # MFCC treatment
    mfcc = librosa.feature.mfcc(y, n_fft = n_fft, hop_length = hop_length, n_mfcc=13)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()
        
    # Get statistics from the vectors
    def get_moments(descriptors):
        result = {}
        for k, v in descriptors.items():
            result['{}_max'.format(k)] = np.max(v)
            result['{}_min'.format(k)] = np.min(v)
            result['{}_mean'.format(k)] = np.mean(v)
            result['{}_std'.format(k)] = np.std(v)
            result['{}_kurtosis'.format(k)] = kurtosis(v)
            result['{}_skew'.format(k)] = skew(v)
        return result
    
    dict_agg_features = get_moments(features)
    dict_agg_features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]
    
    return dict_agg_features


"""
@description: Method to split a song into multiple songs using overlapping windows
"""
def splitsongs(X, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = 33000
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return np.array(temp_X)

"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    # np.array([librosa.power_to_db(s, ref=np.max) for s in list(tsongs)])
    return np.array(list(tsongs))

def make_dataset_dl(song):
    # Convert to spectrograms and split into small windows
    signal, sr = librosa.load(song, sr=None)

    # Convert to dataset of spectograms/melspectograms
    signals = splitsongs(signal)

    # Convert to "spec" representation
    specs = to_melspectrogram(signals)

    return specs
#Completed Making DataSet------------------------

def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_genres(x, dict_genres), prob) for x, prob in votes.items()]


def get_genres(key, dict_genres):
    # Transforming data to help on transformation
    labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}

    return tmp_genre[key]
genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}
song='./static/'
model="custom_cnn_2d.h5"
def dl_run(song,model,genres):
            X = make_dataset_dl(song)
            #ss=song.split("Colab Notebooks",1)[1]
            model = load_model(model)
            preds = model.predict(X)
            votes = majority_voting(preds, genres)
            print("{} is a {} song".format(song, votes[0][0]))
            print("most likely genres are: {}".format(votes[:3]))
            return votes[0][0]

#dl_run(song,model,genres)
#======================================================================================================================>>>>>>>>>>>>>>>>>>>
# import pgm
# flake8: noqa
# prog=pgm()
app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///post1.db'

db = SQLAlchemy(app)



class SongPost(db.Model):
      id = db.Column(db.Integer, primary_key=True)
      title=db.Column(db.String(300),nullable=False)
      data=db.Column(db.LargeBinary,nullable=False)
      Genre=db.Column(db.String(100),nullable=False,default='N/A')
     
      def __repr__(self):
          return 'Blog Post' + str(self.id)

db.create_all()

@app.route("/")
def base():
    return render_template("base.html")
@app.route("/index")
def index():
    return render_template("index.html")
@app.route("/upload")
def upload():
    return render_template("upload.html")
@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        filee=request.files['inputFile']
        newFile=SongPost(title=filee.filename,data=filee.read(),Genre='classical') 
        db.session.add(newFile)
        db.session.commit()
        return redirect('/predict')
    else:
        all_posts=SongPost.query.all()
        return render_template('predict.html',posts=all_posts)   
songUrl=''
@app.route('/post/predict/<int:id>',methods=['GET','POST'])
def edit_only(id):
    songList=[]
    post=SongPost.query.get_or_404(id)
    songList.append(post.title)
    songList.append('./static/'+str(post.title))
    global songUrl
    songUrl='./static/'+str(post.title)
    genre=dl_run(songUrl,model,genres)
    songList.append(genre)
    post.Genre=genre
    db.session.commit()
    # song+=post_tile
    # songList.append(song)
    if request.method == "POST":
        post_title=post.title
        post_data=post.data
        # genre=dl_run(songUrl,model,genres)
        # songList.append(genre)
        post.Genre="genre"
        db.session.commit()
        return redirect('/predict')
    else:
        all_posts=SongPost.query.all()
        return render_template('result.html',sendInfo=all_posts)

@app.route('/result')
def result_get():
    all_posts=SongPost.query.all()
    return render_template('result.html',sendInfo=all_posts)

if __name__ == "__main__":
    app.run(debug=True)
