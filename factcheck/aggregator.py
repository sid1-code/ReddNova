def aggregate(clf, evidence):
    if evidence:
        avg_score = sum(e["score"] for e in evidence) / len(evidence)
    else:
        avg_score = 0

    # Rule-based decision using evidence + classifier
    if avg_score > 1.2:
        final_label = "true"
    elif avg_score < 0.3:
        final_label = "unverified"
    else:
        final_label = clf["pred"]

    explanation = (
        f"Classifier suggests: {clf['pred']} (conf={clf['confidence']:.2f})\n"
        f"Evidence score={avg_score:.2f}\n"
        "Top evidence:\n" +
        "\n".join([f"- {e['title']} ({e['url']}) score={e['score']:.2f}" for e in evidence[:3]])
    )

    return {
        "final_label": final_label,
        "evidence_score": avg_score,
        "reason": explanation
    }
