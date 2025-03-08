with open ("week2\week2.txt", "r",encoding='utf-8') as f:
    lines=f.readlines()# read all lines into a list
for line in lines[:10]:
    print(line.strip())
import jieba
from collections import Counter
import re
all_words=[]
for line in lines:
    line=line.strip()
    words=jieba.lcut(line)# cut the line into words, return a list

    valid_words=[word for word in words if  re.match(r'^[\u4e00-\u9fff]+$',word)]# remove the non-word characters
    all_words.extend(valid_words)# add the words to the list
words_count=Counter(all_words)# count the frequency of each word
print(len(words_count))# print the number of unique words
print(words_count.most_common(10))# print the top 10 words
stopwords = [line.strip() for line in open(r'week2\baidu_stopwords.txt', 'r', encoding='utf-8').readlines()]
filtered_words = [word for word in all_words if word not in stopwords]
filtered_words_count = Counter(filtered_words)
print("Number of unique words after removing stopwords:", len(filtered_words_count))
print(filtered_words_count.most_common(10))
# wordcloud visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
cloud_words = dict(filtered_words_count)
font_path = r'C:\Windows\Fonts\simhei.ttf'
wc = WordCloud(font_path=font_path,width=800, 
                      height=400,
                      background_color="white")
wc.generate_from_frequencies(filtered_words_count)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



# Part of Speech Tagging
import jieba.posseg as pseg
words_with_pos = []

for line in lines:
    line = line.strip()
    words = pseg.lcut(line)# cut the line into words, return a list
    words_with_pos.extend(words)# add the words to the list

