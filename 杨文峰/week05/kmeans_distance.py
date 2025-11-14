import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#文件路径+训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()   #防止句子重复
    with open(path,encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence))) #对句子分词并用空格连接后加入集合
    print("获取句子总数：",len(sentences))
    return sentences

#文本向量化
def sentences_to_vector(sentences,model):
    vectors = []
    for sentence in sentences:
        words = sentence.split() #按空格分隔已分词的句子
        vector = np.zeros(model.vector_size) #先创建全零向量，维度和词向量相同
        for word in words:
            try:
                vector += model.wv[word] #词向量累加
            except KeyError:
                vector += np.zeros(model.vector_size) #若有未登录词，用零向量代替
        vectors.append(vector / len(words)) #计算累加后的平均向量，然后加入向量列表
    return np.array(vectors) #返回向量的数组

#计算类内距离
def intra_distance(vectors,labels,centers):
    cluster_distances = {} #字典存储每个聚类的平均距离
    unique_labels = np.unique(labels) #获取唯一的聚类标签

    for label in unique_labels:
        cluster_vectors = vectors[labels == label] #获取当前聚类的所有向量
        center = centers[label] #获取当前聚类的中心向量

        if len(cluster_vectors > 0):
            distances = [np.linalg.norm(vec - center) for vec in cluster_vectors] #计算每个向量到聚类中心的欧几里得距离
            avg_distance = np.mean(distances)  #计算平均距离
        else:
            avg_distance = float('inf') #如果没有向量，记为无穷大
        cluster_distances[label] = avg_distance #存储该聚类的平均距离

    return cluster_distances

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型文件
    sentences = load_sentence(r"titles.txt") #加载标题文本
    vectors = sentences_to_vector(sentences,model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：",n_clusters)
    kmeans = KMeans(n_clusters) #创建KMeans聚类的对象
    kmeans.fit(vectors) #对向量 进行聚类训练

    sentence_label_dict = defaultdict(list)  #创建默认字典存储分类结果
    for sentence,label in zip(sentences,kmeans.labels_):
        sentence_label_dict[label].append(sentence) #在对应标签下加入句子

    cluster_distances = intra_distance(vectors,kmeans.labels_,kmeans.cluster_centers_)

    sorted_clusters = sorted(cluster_distances.items(),key=lambda x:x[1])[:20] #取平均最小的20个聚类出来

    print("类内距离最短的前20个分类结果：")
    print("=" * 50)

    for label,avg_distance in sorted_clusters:
        print(f"cluster{label} (平均类内距离：{avg_distance:.4f})")
        cluster_sentnces = sentence_label_dict[label]
        for i in range(min(10, len(cluster_sentnces))):
            print(cluster_sentnces[i].replace(" ", ""))
        print("-" * 30)

if __name__ == "__main__":
    main()
