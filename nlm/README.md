Steps to prepare the training and test data.

1. wget http://www.statmt.org/wmt14/training-monolingual-europarl-v7/europarl-v7.en.gz
2. gunzip europarl-v7.en.gz
3. ~/code/cdec/corpus/tokenize-anything.sh < europarl-v7.en | ~/code/cdec/corpus/lowercase.pl >
   europarl-v7.en.lc-tok
4. head -n 2000000 europarl-v7.en.lc-tok > training.en
5. tail -n +2000000 europarl-v7.en.lc-tok > test.en
6. sh ~/code/oxlm/scripts/countcutoff.sh training.en 2
7. python ~/code/oxlm/scripts/preprocess-corpus.py -i training.en,test.en -o training.unk.en,test.unk.en -v vocab
8. ~/code/brown-cluster/wcluster --c 256 --text training.unk.en --threads 10 --output_dir clusters 
