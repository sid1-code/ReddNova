def extract_claim_from_post(title, body):
    if body.strip():
        first_sentence = body.strip().split(".")[0]
        return f"{title}. {first_sentence}"
    return title
