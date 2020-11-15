from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import networkx as nx
import pylab as plt

open_toi = open('/Users/aj/Documents/TOI_data3.txt',encoding='utf-8',mode='r+')
read_toi = open_toi.read()

#Tokenzation
sen_token = PunktSentenceTokenizer()
tokens = sen_token.tokenize(read_toi)

#TF-IDF
matrix = CountVectorizer(stop_words=None).fit_transform(tokens)
print("transform matrix:\n")
print(matrix)
norm = TfidfTransformer().fit_transform(matrix)

print("normalized:\n",norm)
print("\n")
print("normalizer.T:\n",norm.T)
print("\n")

#similarity between sentences
similarity = norm * norm.T
print("similarity graph:\n")
print(similarity)
print("similarity Matrix:\n")
print(similarity.toarray())
print("\n")

#ploting similarity graph
graph = nx.from_scipy_sparse_matrix(similarity)

for i in range(len(similarity.toarray())):
    for j in range(len(similarity.toarray())):
        if i != j:
            kkk = graph.add_edge(i,j,weight=similarity.toarray()[i][j])
            nx.get_edge_attributes(graph,kkk)


pos = nx.spring_layout(graph)
nx.draw(graph,pos)
nx.draw_networkx_edge_labels(graph,pos,labels=nx.get_edge_attributes(graph,kkk))
plt.show()
nx.draw(graph,with_labels=True)
plt.show()

#score of each sentence
scores = nx.pagerank(graph)
print("Score of every sentence :\n")
print(scores)
print("\n")
print("scores & sentence")

jkl = sorted(((scores[i], s) for i, s in enumerate(tokens)),reverse=True)
for i in range(len(jkl)):
    print(jkl[i],'\n')

print("Summary created by TextRank1 function :\n")

for x in jkl[:3]:
    print(x[1])

print("\n")

print("original text:\n")
print(read_toi)
print("original text length:",len(read_toi))
print("\n")

from gensim.summarization.summarizer import summarize
print("summary created by GENSIM library:\n",summarize(read_toi,ratio=0.4))
