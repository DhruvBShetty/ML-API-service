from sentence_transformers import SentenceTransformer, util
import mysql.connector
from queue import PriorityQueue

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="moviedb"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM movie_info")

myresult = mycursor.fetchall()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences=[]
rcmid=[]
for i in myresult:
    sentences.append(i[2])
    rcmid.append(i[0])

embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
simlist=similarities.tolist()

recom=[]
for i in range(len(simlist)):
    for j in range(len(simlist[i])):
        simlist[i][j]=(simlist[i][j],rcmid[j])

for i in simlist:
    i.sort(reverse=True)
    recom.append(i[1:6])
    

for i in range(len(recom)):
    for j in range(len(recom[i])):
        mycursor.execute("INSERT INTO recommend (mid, rid, score) VALUES ({},{},{})".format(rcmid[i],recom[i][j][1],recom[i][j][0]))
        mydb.commit()
    
    


        

    
