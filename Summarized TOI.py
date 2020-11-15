from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import urllib.request
import bs4 as bs
from newspaper import Article

#bs4
#grabbing online textual data
sause = urllib.request.urlopen('https://timesofindia.indiatimes.com/india').read()
soup = bs.BeautifulSoup(sause,"html.parser")
print(soup.title.text)

all_urls = []

data = soup.find_all('ul',attrs={'class':'list5 clearfix'})

for div in data:
    links = div.find_all('a')
    for a in links:
         all_urls.append("https://timesofindia.indiatimes.com/india" + a['href'])


#newspaper

for i in all_urls[:9]:
    toi_news = Article(i,langauge='en')
    toi_news.download()
    toi_news.parse()
    toi_news.nlp()
    output = open("/Users/aj/Documents/TOI_data2.txt", "a", encoding='utf-8')
    output.write(toi_news.title)
    output.write('\n')
    output.write(toi_news.text)
    output.write('\n')

#NLP
a = open('/Users/aj/Documents/TOI_data2.txt',encoding='utf-8',mode='r+')
b = a.read()

#data cleaning
for char in '-,{}+=*!@#$%^&/:<>.’()""“”``\n':
    b = b.replace(char,' ')

b = b.replace('said',' ')
b = b.lower()

#word tokenization
c = word_tokenize(b)
e = []

#word stamming
ps = PorterStemmer()
sw = set(stopwords.words("english"))

c = FreqDist(c)
c = c.most_common()

jj = [x[0] for x in c]


for k in jj:
    if k not in sw:
        e.append(k)

arer = open('/Users/aj/Documents/TOI_data2.txt',encoding='utf-8',mode='r+')
rkk = arer.read()

d = sent_tokenize(rkk)

five = e[:4]

cou = 1

#pick up sentences by their frequency distribution.
for x in d:
    for q in five:
        if q in x:
            print(cou,x)
            cou = cou + 1
            break

print(e[:6])
