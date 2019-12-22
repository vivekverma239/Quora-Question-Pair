mkdir data
mkdir data/glove
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -O data/quora.txt
wget http://nlp.stanford.edu/data/glove.6B.zip -O data/glove/glove.6B.zip 
cd data/glove/
unzip glove.6B.zip