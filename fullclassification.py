import praw
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reddit = praw.Reddit(
    client_id="CWcEE_-6-bz21mUIv6pocQ",
    client_secret="1AAn-dhjzKXqT0Yc9W5ia7FuLEU2Fw",
    user_agent="web:com.example.myredditapp:v1.2.3 (by u/naasirfqi)",

)

# print(device)
# print(reddit.read_only)
# subreddit = reddit.subreddit('Gunners')
# post = list(subreddit.hot(limit=1))[0]
# post.comments.replace_more(limit=0)
# comments = list(post.comments)
# list_comms = []
# for c in comments:
#     list_comms.append(c.body)

list_comms = ['i hate this so much', 'im so angry', 'im nervous about this', 'i was so surprised', 'i am ever feeling nostalgic about the fireplace i will know that it is still on the property']

model_ckpt = "/Users/naasirfarooqi/sentanalysis/distilbert-base-uncased-finetuned-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = 6
def tokenize(batch):
    return tokenizer.batch_encode_plus(batch, padding=True, truncation=True, return_tensors='pt')
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)

tokenized_inputs = tokenize(list_comms)
input_ids = tokenized_inputs["input_ids"].to(device)
attention_mask = tokenized_inputs["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]

y_preds = np.argmax(logits, axis=1)
print(y_preds)


