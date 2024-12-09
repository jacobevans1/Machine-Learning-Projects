# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Read the new dataset
df = pd.read_csv('FinancialPhraseBank.csv', names=['status', 'statement'], encoding='ISO-8859-1')

# Display basic info
print("Dataset Overview:\n", df.head())
print("Missing values:\n", df.isnull().sum())
print("Unique statuses:\n", df['status'].unique())

# Fill missing values in the statement column if necessary
df['statement'] = df['statement'].fillna('')

# Clean the text data
stopwordss = stopwords.words('english')
lem = WordNetLemmatizer()

def clean(line):
    line = line.lower()
    line = re.sub(r'\d+', '', line)  # Remove numbers
    line = re.sub(r'[^a-zA-Z0-9\s]', '', line)  # Remove special characters
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)  # Remove punctuation
    words = [word for word in line.split() if word not in stopwordss]  # Remove stopwords
    words = [lem.lemmatize(word) for word in words]  # Lemmatize words
    return ' '.join(words)

df['statement'] = df['statement'].apply(clean)
print("Cleaned Data:\n", df.head())

# Encode labels
le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])
print("Label classes:", list(le.classes_))

# Prepare text data and labels
x = df['statement']
y = df['status']

# Tokenize text
tokenizer = Tokenizer(oov_token='<unk>', num_words=2500)
tokenizer.fit_on_texts(x.values)
data_x = tokenizer.texts_to_sequences(x.values)

# Pad sequences
max_len = 42
data_x = pad_sequences(data_x, maxlen=max_len, padding='post', truncating='post')

# Convert labels to one-hot encoding
y = pd.get_dummies(y).values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_x, y, test_size=0.2, random_state=0, stratify=y)


vocab_size = len(tokenizer.word_index) + 1
embedding_size = 50
latent_size = 200


# Define a function to build, train, and evaluate models
def train_and_evaluate(embedding_size, filters, kernel_size, latent_size, dropout_rate, batch_size, epochs):
    # Define the model
    model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        Dropout(dropout_rate),
        Conv1D(filters, kernel_size, activation='relu'),
        MaxPooling1D(pool_size=4),
        LSTM(latent_size),
        Dense(3, activation='softmax')  # 3 classes: positive, negative, neutral
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                        verbose=1)

    # Evaluate the model
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {score}")
    print(f"Test Accuracy: {acc}")

    return history, acc, score


# Experiment 1: Baseline
history1, acc1, loss1 = train_and_evaluate(
    embedding_size=50, filters=64, kernel_size=5, latent_size=200, dropout_rate=0.25, batch_size=128, epochs=10
)

# Experiment 2: Larger embedding and latent size
history2, acc2, loss2 = train_and_evaluate(
    embedding_size=100, filters=128, kernel_size=3, latent_size=300, dropout_rate=0.3, batch_size=64, epochs=12
)

# Experiment 3: Smaller latent size and different dropout
history3, acc3, loss3 = train_and_evaluate(
    embedding_size=50, filters=32, kernel_size=7, latent_size=100, dropout_rate=0.2, batch_size=256, epochs=8
)

# Compare Results
results = pd.DataFrame({
    'Experiment': ['Baseline', 'Larger Embedding/Latent Size', 'Smaller Latent Size'],
    'Test Accuracy': [acc1, acc2, acc3],
    'Test Loss': [loss1, loss2, loss3]
})

print("\nComparison of Results:\n", results)


# Plot training history for all experiments
def plot_history(history, label):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, val_acc, label=f"{label} - Validation")
    plt.plot(epochs, acc, linestyle='dashed', label=f"{label} - Training")


plt.figure(figsize=(12, 6))
plot_history(history1, "Baseline")
plot_history(history2, "Larger Embedding/Latent Size")
plot_history(history3, "Smaller Latent Size")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()