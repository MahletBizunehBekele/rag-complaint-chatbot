from rag_pipeline import ask

questions = [
    "Why are customers unhappy with credit cards?",
    "What complaints are common in savings accounts?",
    "What issues occur with money transfers?",
    "What fraud complaints are reported?",
    "Compare credit cards and personal loans."
]

for q in questions:
    answer, sources = ask(q)

    print("\n" + "="*80)
    print("QUESTION:", q)
    print("\nANSWER:")
    print(answer)
    print("\nSOURCE:")
    print(sources[0][:300])