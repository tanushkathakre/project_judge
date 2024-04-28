import transformers
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load the judgments and summaries from the CSV file


def load_judgments_and_summaries(csv_file):
    judgments = []
    summaries = []
    with open(csv_file, "r", encoding="utf-8") as file:
        for line in file:
            data = line.strip().split(",", 1)  # Limit split to 1 to avoid further splitting
            if len(data) == 2:  # Check if there are exactly two values
                judgment, summary = data
                judgments.append(judgment)
                summaries.append(summary)
    return judgments, summaries

# Function to generate summary for a given text


def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=750, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


judgments, summaries = load_judgments_and_summaries("summary.csv")

# Example usage
input_text = ("""Complainant's Account:
The complainant, Mr. Ramesh Sharma, reports that his 6-year-old daughter, Sunita Sharma, was abducted on the evening of April 30th, 2024, while returning home from school. According to Mr. Sharma, Sunita was walking home from her school bus stop when a white van pulled up beside her. Two masked individuals forcefully grabbed Sunita and dragged her into the van before speeding away. Mr. Sharma immediately reported the incident to the local police station and has been actively cooperating with the investigation ever since. He claims to have no knowledge of the abductors' identities or motives.

Defendant's Account:
The defendant, Mr. Rajesh Singh, denies any involvement in the alleged kidnapping of Sunita Sharma. According to Mr. Singh, he was at his workplace, a construction site, during the time of the incident. He states that he has never met Sunita Sharma or her family and has no reason to harm or abduct her. Mr. Singh asserts that he is being falsely accused and demands a thorough investigation to clear his name.

Witness Testimony:
An eyewitness, Mrs. Geeta Verma, corroborates Mr. Sharma's account of the abduction. Mrs. Verma, who lives near the school bus stop, claims to have seen the white van and the masked individuals forcibly taking Sunita Sharma. She immediately informed the authorities and has provided a detailed description of the van and the suspects.

Investigation Findings:
The investigation has revealed CCTV footage from nearby cameras, which captures the white van and the masked individuals at the scene of the abduction. The police are currently analyzing the footage to identify the suspects and gather further evidence.

Legal Proceedings:
Mr. Ramesh Sharma has filed a formal complaint with the police, accusing Mr. Rajesh Singh of kidnapping his daughter, Sunita Sharma. The police have initiated an investigation into the matter and are working diligently to locate and rescue Sunita. Mr. Singh maintains his innocence and awaits a fair trial to prove his innocence.
""")
generated_summary = generate_summary(input_text)
print("Generated Summary:", generated_summary)
