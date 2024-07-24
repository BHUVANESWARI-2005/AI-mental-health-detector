import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

working_dir = os.getcwd()
json_file_path = os.path.join(working_dir, r'C:\mental\primate_dataset.json')
model_file_path = os.path.join(working_dir, r'C:\mental\model.json')
vectorizer_file_path = os.path.join(working_dir, r'C:\mental\vectorizer.pkl')
test_results_file_path = os.path.join(working_dir, r'C:\mental\test_results.txt')


with open(json_file_path, 'r') as file:
    data = json.load(file)


df = pd.DataFrame(data)


df['annotations'] = df['annotations'].apply(lambda x: [label for sublist in x for label in sublist])


mlb = MultiLabelBinarizer()
labels_binary = mlb.fit_transform(df['annotations'])


train_data, test_data, train_labels, test_labels = train_test_split(df['post_text'], labels_binary, test_size=0.2,
                                                                    random_state=42)


model = make_pipeline(TfidfVectorizer(), LabelPowerset(classifier=LogisticRegression()))


model.fit(train_data, train_labels)


predictions = model.predict(test_data)


accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
f1 = f1_score(test_labels, predictions, average='weighted')


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


with open(model_file_path, 'w') as model_file:
    # Saving classifier parameters
    model_params = model.named_steps['labelpowerset'].get_params()
    model_params['classifier'] = model_params['classifier'].get_params()
    json.dump(model_params, model_file)

with open(vectorizer_file_path, 'wb') as vectorizer_file:
    pickle.dump(model.named_steps['tfidfvectorizer'], vectorizer_file)


with open(test_results_file_path, 'w', encoding='utf-8') as results_file:
    for text, true_label, predicted_label in zip(test_data, test_labels, predictions):
        results_file.write(f"Text: {text}\nTrue Label: {true_label}\nPredicted Label: {predicted_label}\n\n")