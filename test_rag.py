from src.rag_pipeline import ask

answer, sources = ask(
    "Why are customers unhappy with credit cards?"
)

print("\nANSWER:")
print(answer)

print("\nSOURCE:")
print(sources[0])