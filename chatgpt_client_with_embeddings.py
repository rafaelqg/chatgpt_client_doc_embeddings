#author: Rafael Queiroz Gonçalves
import openai
import os.path
import pandas as pd
import PyPDF2
from openai.embeddings_utils import cosine_similarity

def generateEmbeddingFromFile(filePath):
  embeddingFileName = filePath.replace(".", "") + "_embedding.txt"
  textFileName = filePath.replace(".", "") + ".txt"
  isEmbeddedFileAlready = os.path.isfile(embeddingFileName)
  if not isEmbeddedFileAlready:
    pdfURL = filePath
    # creating a pdf reader object
    reader = PyPDF2.PdfReader(pdfURL)
    # printing number of pages in pdf file
    # getting a specific page from the pdf file
    text = ""
    for i in range(0, len(reader.pages)):
      page = reader.pages[i]
      pageText = page.extract_text()
      text += pageText
    #print(text)

    f = open(textFileName, "w")
    f.write(text)
    f.close()

    response = openai.Embedding.create(input=text,
                                       model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    f = open(embeddingFileName, "w")
    f.write(str(embedding))
    f.close()


def buildEmbbedingsArray(filesToProcess):
  dataset = []
  for filePath in filesToProcess:
    embeddingFileName = filePath.replace(".", "") + "_embedding.txt"
    textFileName = filePath.replace(".", "") + ".txt"
    f = open(embeddingFileName, "r")
    embedding = f.read()
    f = open(textFileName, "r")
    text = f.read()
    embedding = embedding.replace("[", "").replace("]", "").split(", ")
    input_data = []
    for value in embedding:
      input_data.append(float(value))
    dict_entry = {"text": text, "text_embedding": input_data}
    dataset.append(dict_entry)
  return dataset


#code init
api_key = "<your chatgpt api key"
openai.api_key = api_key

filesToProcess = [
  "<file path 01>", "<file path 02>"
]
for filePath in filesToProcess:
  generateEmbeddingFromFile(filePath)
embbedingsDataset = buildEmbbedingsArray(filesToProcess)
#print(embbedingsDataset)

#2) Prepare embedding for user question
user_question = "Como o uso do sistema SEI foi regulamentado no TCE/SC?"
response = openai.Embedding.create(input=user_question,
                                   model="text-embedding-ada-002")
embeddings_customer_question = response['data'][0]['embedding']
#print("Embedding question", embeddings_customer_question)

#3) Find similarities based on embeddings
pandas_dataframe = pd.DataFrame(embbedingsDataset)
pandas_dataframe['files_search'] = pandas_dataframe.text_embedding.apply(
  lambda x: cosine_similarity(x, embeddings_customer_question))
pandas_dataframe = pandas_dataframe.sort_values('files_search',
                                                ascending=False)
pandas_dataframe = pandas_dataframe.head(3)
#print("top 3 similarity", pandas_dataframe)
context = pandas_dataframe.at[0, 'text']
#print("Most similar content", context, type(context))

#4) ask o chatGPT with  proper context
messages = [{
  "role": "system",
  "content": context
}, {
  "role": "user",
  "content": user_question
}]
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=messages)
print("Question", user_question)
print("ChatGPT answer", completion.choices[0].message.content)
