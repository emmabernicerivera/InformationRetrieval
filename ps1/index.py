import os
import glob
import string
from math import log10
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# initialize porter stemmer 
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# initialize indices
doc_index = {}
posting_index = {}

stoplist = open("./stoplist.txt", "r").read().split()

def createIndex(pathname):
	files = glob.glob(pathname)

	for filename in files:
		file = open(filename, "r").read()
		# remove punctuation 
		file = file.translate(str.maketrans('', '', string.punctuation))
		# split by whitespace 
		file = file.split()
		# lowercase all letters and remove stop words
		file = [word.lower() for word in file]
		file = [word for word in file if word not in stoplist]
		# perform stemming 
		file = [ps.stem(word) for word in file]
		# perform lemmitization 
		file = [lemmatizer.lemmatize(word) for word in file]
		# populate indices 
		name = os.path.basename(filename)
		doc_index[name] = len(file)

		# calculate the term frequency for all words in file
		tf = {}
		for word in file: 
			if word in tf:
				tf[word] = tf[word] + 1
			else:
				tf[word] = 1

		# tf for each word in each doc
		for word in tf:

			if word in posting_index:
				posting_index[word][name] = tf[word]
			else:
				posting_index[word] = {name: tf[word]}

	# add number of documents with each word
	for word in posting_index:
		posting_index[word] = (len(posting_index[word]), posting_index[word])


def termLookup(query, pathname):
	files = glob.glob(pathname)
	
	# remove punctuation 
	query = query.translate(str.maketrans('', '', string.punctuation))
	# split by whitespace 
	query = query.split()
	# lowercase all letters and remove stop words
	query = [word.lower() for word in query]
	query = [word for word in query if word not in stoplist]
	# perform stemming 
	query = [ps.stem(word) for word in query]
	# perform lemmitization 
	query = [lemmatizer.lemmatize(word) for word in query]

	if len(query) == 0:
		print("No Match")
		return

	for word in query: 
		if word not in posting_index:
			print(f"No Match for {word}")
			continue
		else:
			print(f'TF-IDF weights for {word}:')
			for file in files:
				name = os.path.basename(file)
				tf = 0
				if name in posting_index[word][1]:
					tf = posting_index[word][1][name] / doc_index[name]

				idf = 1 + log10(len(files) / posting_index[word][0] + 1)
				print(f'{name}: {tf * idf}')


pathname = input("Which files would you like to process? Example: ./dataset1/* \n --> ")
createIndex(pathname)
# createIndex("./dataset1/*")

query = input("Search: ")
termLookup(query, pathname)
# termLookup(query, "./dataset1/*")


