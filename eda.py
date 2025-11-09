"""
Information about dataset:
Features:
unique_id: A unique identifier for each entry.
Statement: The textual data or post.
Mental Health Status: The tagged mental health status of the statement.
"""

import re
import string
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')   
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the given DataFrame by providing key statistics for each column.

    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.

    Returns:
    pd.DataFrame: A summary DataFrame containing statistics for each column.
    """
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Unique Values': df.nunique(),
        'Mean': df.select_dtypes(include=['number']).mean(),
        'Median': df.select_dtypes(include=['number']).median(),
        'Min': df.select_dtypes(include=['number']).min(),
        'Max': df.select_dtypes(include=['number']).max(),
        'Std Dev': df.select_dtypes(include=['number']).std()
    })

    return summary


data = pd.read_csv('data/Combined_data.csv')
data['statement'] = data['statement'].astype(str)
print(summarize_dataframe(data))


"""
a. Load & Inspect

Check for missing or duplicate Statement entries.

Examine class distribution and total record count.

Compute text length statistics (tokens per post, characters).
"""
missing_data = data['statement'].isnull().sum()
duplicate_data = data['statement'].duplicated().sum()
class_distribution = data['status'].value_counts()
total_records = len(data)

print("Missing data:")
print(missing_data)

print("Duplicate data:")
print(duplicate_data)

print("Class distribution:")
print(class_distribution)

print("Total records:")
print(total_records)




"""
b. Cleaning

Lowercasing, remove URLs, emojis, punctuation.

Expand contractions (e.g., “don’t → do not”).

Optional: lemmatize for LDA pipeline.
"""

def clean_text(text: str, lemmatize: bool = False) -> str:
    """
    Cleans the input text by lowercasing, removing URLs, emojis, punctuation,
    expanding contractions, and optionally lemmatizing.

    Parameters:
    text (str): The text to clean.
    lemmatize (bool): Whether to lemmatize the text.

    Returns:
    str: The cleaned text.
    """

    # Lowercase
    # print("text: ", text)
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize if required
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)
data['cleaned_statement'] = data['statement'].apply(lambda x: clean_text(x, lemmatize=True))
print(data[['statement', 'cleaned_statement']].head())  
                    


"""c. Exploration

Bar chart of label frequencies (to reveal imbalance).

Word clouds / top n-grams per category.

TF-IDF keyword comparison across classes."""

# Bar chart of label frequencies
label_counts = data['status'].value_counts()
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar')
plt.title('Label Frequencies')
plt.xlabel('Mental Health Status')
plt.ylabel('Frequency')
plt.show()


