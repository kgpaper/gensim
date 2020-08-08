from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import re 

nsmc = pd.read_csv('ratings_train.txt', sep='\t')

#한글로 이루어진 단어 추출하는 함수
def find_hangul(text):
    return re.findall(r'[ㄱ-ㅎ가-힣]+', text)

data = nsmc[nsmc['document'].notnull()]['document'].map(find_hangul)

model = FastText(size=16)
model.build_vocab(sentences=data)
model.train(
    sentences=data,
    epochs=5,
    total_examples=model.corpus_count,
    total_words=model.corpus_total_words
)

ft = model
df = nsmc[nsmc['document'].notnull()]

#훈련용 데이터와 테스트용 데이터 분할
doc_train, doc_test, y_train, y_test = train_test_split(df['document'], df['label'], test_size=0.2, random_state=42)

x_train = np.zeros((1000, 16))

for i, doc in enumerate(doc_train.iloc[:1000]):
    vs = [ft.wv[word] for word in find_hangul(doc)]
    if vs:
        x_train[i,] = np.mean(vs, axis=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train.values[:1000], epochs=1)
model.save('nsmc.fasttext')