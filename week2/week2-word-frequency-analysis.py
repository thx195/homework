# Read data from file and print first 10 lines
with open("week2\week2.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()  # read all lines into a list
print("=== First 10 lines ===")
for line in lines[:10]:
    print(line.strip())

# Tokenization and word frequency analysis
import jieba
from collections import Counter
import re

all_words = []
for line in lines:
    line = line.strip()
    # Tokenize with jieba and filter non-Chinese characters
    words = jieba.lcut(line)  # cut the line into words, return a list
    valid_words = [word for word in words if re.match(r'^[\u4e00-\u9fff]+$', word)]
    all_words.extend(valid_words)

words_count = Counter(all_words)  # count the frequency of each word
print("\n=== Before removing stopwords ===")
print("Unique words count:", len(words_count))
print("Top 10 frequent words:", words_count.most_common(10))

# Load stopwords and filter
stopwords = [line.strip() for line in open(r'week2\baidu_stopwords.txt', 'r', encoding='utf-8').readlines()]
filtered_words = [word for word in all_words if word not in stopwords]
filtered_words_count = Counter(filtered_words)
print("\n=== After removing stopwords ===")
print("Unique words count:", len(filtered_words_count))
print("New top 10 frequent words:", filtered_words_count.most_common(10))

# Generate word cloud visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt

font_path = r'C:\Windows\Fonts\simhei.ttf'
wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white")
wc.generate_from_frequencies(filtered_words_count)

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# Part 6: POS tagging analysis (Enhanced version)
import jieba.posseg as pseg

words_with_pos = []
for line in lines:
    line = line.strip()
    # Clean text before POS tagging using same rules
    cleaned_line = re.sub(r'[^\u4e00-\u9fff]', ' ', line)  # Remove non-Chinese characters
    pairs = pseg.lcut(cleaned_line)  # Get POS tags
    
    # Apply same filtering as main processing
    valid_pairs = [
        (word, flag) for word, flag in pairs 
        if re.match(r'^[\u4e00-\u9fff]+$', word) and word not in stopwords
    ]
    words_with_pos.extend(valid_pairs)

pos_counts = Counter([flag for word, flag in words_with_pos])
print("\n=== POS Tag Frequency ===")
print("Top 10 POS tags:", pos_counts.most_common(10))

# Part 7: Bigram analysis (Sentence-level processing)
from nltk import bigrams

bigram_list = []
for line in lines:
    line = line.strip()
    # Process each sentence independently
    words = jieba.lcut(line)
    valid_words = [word for word in words if re.match(r'^[\u4e00-\u9fff]+$', word) and word not in stopwords]
    bigram_list.extend(list(bigrams(valid_words)))  # Generate bigrams per sentence

bigram_counts = Counter(bigram_list)
print("\n=== Bigram Frequency ===")
print("Top 10 bigrams:", bigram_counts.most_common(10))

# Part 8: Text vectorization and similarity calculation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create document-term matrix
corpus = []
for line in lines:
    line = line.strip()
    words = jieba.lcut(line)
    valid_words = [word for word in words if re.match(r'^[\u4e00-\u9fff]+$', word) and word not in stopwords]
    corpus.append(' '.join(valid_words))  # Convert words to space-separated string

# Use top 100 words as features
top_features = [word for word, count in filtered_words_count.most_common(100)]
vectorizer = CountVectorizer(vocabulary=top_features)
vectors = vectorizer.fit_transform(corpus)

# Calculate similarity between first two sentences
if len(corpus) >= 2:
    similarity = cosine_similarity(vectors[0], vectors[1])
    print("\n=== Sentence Similarity ===")
    print(f"Similarity between sentence 0 and 1: {similarity[0][0]:.4f}")
else:
    print("\nNot enough sentences for similarity calculation")