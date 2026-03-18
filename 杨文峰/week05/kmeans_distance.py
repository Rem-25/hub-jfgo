import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

from torch.signal.windows import cosine


#加载模型、加载句子、文本向量化、用模型聚类训练、计算类内距离

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
        print("已获取句子数量：%d" %len(sentences))
        return sentences

def sentence_2_vec(sentences, model):
    vecs = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vecs.append(vector / len(words))
    return np.array(vecs)

def main():
    model = Word2Vec.load('model.w2v')
    sentences = load_sentence('titles.txt')
    vecs = sentence_2_vec(sentences, model)

    clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：%d" %clusters)
    kmeans = KMeans(clusters)
    kmeans.fit(vecs)

    sentences_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentences_label_dict[label].append(sentence)

    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vecs[vector_index]
        center = kmeans.cluster_centers_[label]
        distance = cosine_distance(vector, center)
        density_dict[label].append(distance)
    for label, distance_list in density_dict.items():
        density_dict[label] = np.mean(distance_list)
    density_order = sorted(density_dict.items(), key=lambda x:x[1], reverse=True)

    for label, distance_avg in density_order:
        print("聚类序号：%s, 平均距离：%f" %(label, distance_avg))
        sentences = sentences_label_dict[label]
        for i in range(min(20, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("-"*10)

def cosine_distance(vector1, vector2):
    vector1 = vector1 / np.sqrt(np.sum(np.square(vector1)))
    vector2 = vector2 / np.sqrt(np.sum(np.square(vector2)))
    return np.sum(vector1 * vector2)

def eculid_distance(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2)))

if __name__ == "__main__":
    main()
