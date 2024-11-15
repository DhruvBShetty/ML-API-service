from sentence_transformers import SentenceTransformer, util
sentences = ["I'm happy", "I'm full of happiness"]

model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

a=util.pytorch_cos_sim(embedding_1, embedding_2)
print(a[0][0])