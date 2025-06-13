from chromadb import HttpClient
client = HttpClient(host="localhost", port=8001)
collection = client.get_or_create_collection("multimodal")
print("📊 Total documentos:", collection.count())
