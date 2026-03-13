from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_distilbert_model():
    """
    Load pretrained DistilBERT model for sentiment classification.
    """

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    return tokenizer, model
