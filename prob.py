import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import string  # Add this import statement
# Load data
x_train_df = pd.read_csv("X_train.csv")
y_train_df = pd.read_csv("y_train.csv")
x_test_df = pd.read_csv("X_test.csv")
y_test_df = pd.read_csv("y_test.csv")

# Preprocessing function


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply preprocessing to 'Facts' column


x_train_df['Facts'] = x_train_df['Facts'].apply(preprocess_text)
x_test_df['Facts'] = x_test_df['Facts'].apply(preprocess_text)

# Define and train the model
model = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression()
)

model.fit(x_train_df['Facts'], y_train_df['winner_index'])

# Predict on test set
predictions = model.predict(x_test_df['Facts'])

# Evaluate the model
accuracy = accuracy_score(y_test_df['winner_index'], predictions)
print(f"Accuracy: {accuracy}")


def predict_winning_probability(petitioner, respondent, facts):
    # Preprocess the input text
    processed_facts = preprocess_text(facts)
    input_text = f"{petitioner} {respondent} {processed_facts}"
    # Use the trained model to predict probabilities
    probabilities = model.predict_proba([input_text])[0]
    return probabilities

# Example inputs


petitioner = "Mr. Ramesh Sharma"
respondent = "Mr. Rajesh Singh"
facts = """Complainant's Account:
The complainant, Mr. Ramesh Sharma, reports that his 6-year-old daughter, Sunita Sharma, was abducted on the evening of April 30th, 2024, while returning home from school. According to Mr. Sharma, Sunita was walking home from her school bus stop when a white van pulled up beside her. Two masked individuals forcefully grabbed Sunita and dragged her into the van before speeding away. Mr. Sharma immediately reported the incident to the local police station and has been actively cooperating with the investigation ever since. He claims to have no knowledge of the abductors' identities or motives.

Defendant's Account:
The defendant, Mr. Rajesh Singh, denies any involvement in the alleged kidnapping of Sunita Sharma. According to Mr. Singh, he was at his workplace, a construction site, during the time of the incident. He states that he has never met Sunita Sharma or her family and has no reason to harm or abduct her. Mr. Singh asserts that he is being falsely accused and demands a thorough investigation to clear his name.

Witness Testimony:
An eyewitness, Mrs. Geeta Verma, corroborates Mr. Sharma's account of the abduction. Mrs. Verma, who lives near the school bus stop, claims to have seen the white van and the masked individuals forcibly taking Sunita Sharma. She immediately informed the authorities and has provided a detailed description of the van and the suspects.

Investigation Findings:
The investigation has revealed CCTV footage from nearby cameras, which captures the white van and the masked individuals at the scene of the abduction. The police are currently analyzing the footage to identify the suspects and gather further evidence.

Legal Proceedings:
Mr. Ramesh Sharma has filed a formal complaint with the police, accusing Mr. Rajesh Singh of kidnapping his daughter, Sunita Sharma. The police have initiated an investigation into the matter and are working diligently to locate and rescue Sunita. Mr. Singh maintains his innocence and awaits a fair trial to prove his innocence.
"""
# Predict and display result
winning_probabilities = predict_winning_probability(petitioner, respondent, facts)
print(f"Winning Probability for Petitioner: {winning_probabilities[1]}")
print(f"Winning Probability for Respondent: {winning_probabilities[0]}")
