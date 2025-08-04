from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Example log keys (treat each as a "document" of words)
log_keys = [
    "user login successful",
    "user logout completed", 
    "database connection established",
    "database query executed",
    "database connection closed",
    "file read operation",
    "file write operation", 
    "authentication failed",
    "user login attempt",
    "network connection timeout",
    "http request received",
    "sql query executed",
    "file access denied",
    "user session expired"
]

vectorizer = CountVectorizer(stop_words='english')
log_word_matrix = vectorizer.fit_transform(log_keys)
feature_names = vectorizer.get_feature_names_out()

# Apply LDA with 4 topics (categories)
n_topics = 4
lda = LatentDirichletAllocation(
    n_components=n_topics, 
    random_state=42,
    max_iter=100
)

lda.fit(log_word_matrix)

def display_topics(model, feature_names, no_top_words=5):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[-no_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic_{topic_idx}"] = top_words
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    return topics

discovered_topics = display_topics(lda, feature_names)
# print("\nDiscovered Topics:")
# for topic, words in discovered_topics.items():
#     print(f"{topic}: {', '.join(words)}")


def classify_log_key(log_key, model, vectorizer, topic_names=None):
    # Convert log key to vector
    log_vector = vectorizer.transform([log_key])
    
    # Get topic probabilities
    topic_probs = model.transform(log_vector)[0]
    
    # Find dominant topic
    dominant_topic = np.argmax(topic_probs)
    confidence = topic_probs[dominant_topic]
    
    if topic_names:
        return topic_names[dominant_topic], confidence
    else:
        return f"Topic_{dominant_topic}", confidence

# Example usage
new_log = "database backup completed"
category, confidence = classify_log_key(new_log, lda, vectorizer)
print(f"'{new_log}' -> {category} (confidence: {confidence:.3f})")