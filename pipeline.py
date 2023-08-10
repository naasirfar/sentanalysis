from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd
import praw

reddit = praw.Reddit(
    client_id="not",
    client_secret="your",
    user_agent="business",

)

subreddit = reddit.subreddit('Gunners')
post = list(subreddit.hot(limit=1))[0]
post.comments.replace_more(limit=0)
comments = list(post.comments)
list_comms = []
for c in comments:
    list_comms.append(c.body)


# Change `transformersbook` to your Hub username
labels = {'LABEL_0': 'sadness', 'LABEL_1': 'joy', 'LABEL_2': 'love', 'LABEL_3': 'anger', 'LABEL_4': 'fear', 'LABEL_5':'surprise'}
model_id = "naasirfar/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)
# custom_tweet = "I saw a movie today and it was really good."
# preds = classifier(custom_tweet, return_all_scores=True)


# preds_df = pd.DataFrame(preds[0])
# plt.bar(labels, 100 * preds_df["score"], color='C0')
# plt.title(f'"{custom_tweet}"')
# plt.ylabel("Class probability (%)")
# plt.show()

def label_int2str(entry):
    return labels[entry['label']]

preds = classifier(list_comms)
print(list(map(label_int2str, preds)))
