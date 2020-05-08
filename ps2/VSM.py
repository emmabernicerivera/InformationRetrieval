import os
import glob
import string
import re
import sys
from math import log10, sqrt
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# initialize porter stemmer 
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# vector space model 
# {
# 	docid: {
# 		word: freq
# 	}
# }
docid_to_tf = {}

# {
# 	word: {
# 		doc_id: doc_tf
# 	}
# }
term_to_doc_tf = {}

# {
# 	doc_number: text
# } 
docs = {}

stoplist = open("./stoplist.txt", "r").read().split()

def createIndex(pathname):
	current_docno = ""
	start_line = 0
	lines = open(pathname, "r").read().splitlines()

	with open(pathname, "r") as file:
		for (line_number, line) in enumerate(file):
			if "<DOCNO>" in line:
				current_docno = re.findall(r'<DOCNO> (.*?) </DOCNO>', line)[0]
			if "<TEXT>" in line:
				start_line = line_number
			if "</TEXT>" in line:
				if current_docno in docs:
					docs[current_docno] = docs[current_docno] + " " + " ".join(lines[start_line + 1:line_number])
				else:
					docs[current_docno] = " ".join(lines[start_line + 1:line_number])

	for doc_id in docs:
		doc = docs[doc_id]
		# remove punctuation 
		doc = doc.translate(str.maketrans('', '', string.punctuation))
		# split by whitespace 
		doc = doc.split()
		# lowercase all letters and remove stop words
		doc = [word.lower() for word in doc]
		doc = [word for word in doc if word not in stoplist and not word.isdigit()]
		# perform stemming 
		doc = [ps.stem(word) for word in doc]
		# perform lemmitization 
		doc = [lemmatizer.lemmatize(word) for word in doc]

		docs[doc_id] = doc
		
		docid_to_tf[doc_id] = {}
		for word in doc:
			if word not in docid_to_tf[doc_id]:
				docid_to_tf[doc_id][word] = 1
			else:
				docid_to_tf[doc_id][word] = docid_to_tf[doc_id][word] + 1

		for word in doc:
			if word not in term_to_doc_tf:
				term_to_doc_tf[word] = {}
			
			term_to_doc_tf[word][doc_id] = docid_to_tf[doc_id][word]


	for word in term_to_doc_tf:
		for doc_id in docs:
			if word not in docid_to_tf[doc_id]:
				docid_to_tf[doc_id][word] = 0

	return

def termLookup(query, result_file):
	
	# open query file
	query = open(query, "r").read().splitlines()
	results = open(result_file, "a")
	# {
	# 	query_num: query_text 
	# }
	#intitalize map with unique query num 
	query_list = {}
	for q in query:
		query_num = q.split(' ')[0][:-1]
		query_list[query_num] = ""

	# merge query back into one blob of text 
	query = ' '.join(query)
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
	
	curr_queryno = 0
	for word in query:
		if word in query_list:
			curr_queryno = word
		else:
			query_list[curr_queryno] = query_list[curr_queryno] + " " + word

	# {
	# 	word: freq
	# }
	query_tf = {}
	# {
	# 	doc_id: cos_sim
	# }
	cos_sim_rank  = defaultdict(int)
	cos_sim_rank_sorted = {}
	cnt = 1
	for queryno in query_list:
		for word in term_to_doc_tf:
			query_tf[word] = query_list[queryno].count(word)
		for doc_id in docs:
			cos_sim_rank[doc_id] = dotproduct(query_tf, docid_to_tf[doc_id]) / (magnitude(docid_to_tf[doc_id]) * magnitude(query_tf))
		for doc_id in sorted(cos_sim_rank, key=cos_sim_rank.get, reverse=True):
			cos_sim_rank_sorted[doc_id] = cos_sim_rank[doc_id]
		for doc_id in cos_sim_rank_sorted:
			results.write('%s Q0 %s %d %.19f Exp \n' % (queryno, doc_id, cnt, cos_sim_rank_sorted[doc_id]))
			cnt = cnt + 1
			if cnt == 21:
				cnt = 1
				break

	results.close()
	return

def dotproduct(v1, v2):
	sum = 0
	for key in v1:
		sum = sum + (v1[key] * v2[key])
	return sum

def magnitude(v):
	sum = 0
	for key in v:
		sum = sum + v[key]**2
	return sqrt(sum)

# ./VSM.py <dir-path-to-collection> <query-file> <name-of-results-file> 
createIndex(sys.argv[1])
termLookup(sys.argv[2], sys.argv[3])


