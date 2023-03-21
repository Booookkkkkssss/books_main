# subset df split of train into only negative values of 'sentiment' column
df_neg = train[train['sentiment'] == 'negative']

# Tokenize words
words = df_neg['cleaned_summary'].str.lower().str.cat(sep=' ').split()

# Create a list of additional stopwords
additional_stopwords = ['.']

# Add the additional stopwords to the default set of stopwords
stop_words = set(stopwords.words('english') + additional_stopwords)

# Remove stopwords
words = [word for word in words if word not in stop_words]

# Get frequency distribution of words
freq_dist = nltk.FreqDist(words)

# Get 50 most common words
most_common = freq_dist.most_common(50)

# convert to dataframe for visualizing ease
df_common_words = pd.DataFrame(most_common, columns=['Word', 'Frequency'])

# visual output
df_common_words.head(15)