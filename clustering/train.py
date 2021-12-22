# -*- coding: utf-8 -*-
import sys
import gensim
import sklearn
import numpy as np
import json
import jieba
import re
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from datamanager import DataManager

def loadLyrics(self, filepath):
    dict_stat = {'ReadError': 0}
    lines = []
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # try:
            obj = json.loads(line)
            lyrics = obj["lyrics"]["content"].values()
            lyrics_str = ""
            for lv in lyrics:
                lyrics_str += lv
            print(lyrics_str)
            lyrics_str = re.sub("[a-zA-Z0-9\!\%\[\]\,\。\:\：\-\"\“\”\(\)（）{}\'']", "", lyrics_str)
            lyrics_str = lyrics_str.replace(" ", "")
            words = list(jieba.cut(str(lyrics_str)))
            word_list = []
            for w in words:
                word_list.append(w)
            # filter those with less words in the lyrics
            if len(word_list) > 20:
                obj["lyrics"]["seg"] = word_list
                lines.append(obj)



TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_datasest():
    dm = DataManager()
    list_lyrics = dm.loadLyrics('lyrics1.json')
    x_train = []
    # y = np.concatenate(np.ones(len(docs)))
    for i, lyric in enumerate(list_lyrics):
        try:
            document = TaggededDocument(lyric["lyrics"]["seg"], tags=[i])
            x_train.append(document)
        except:
            print(i)
            # print(type(words))
    return x_train

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)


def train(x_train, size=100, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_dm_lyrics.vec')
    model_dm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model_dm
