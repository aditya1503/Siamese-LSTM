# Siamese-LSTM
Download the word2vec model from
https://code.google.com/archive/p/word2vec/ 
and download the file: GoogleNews-vectors-negative300.bin.gz
Set training=False if you want to load trained weights
Files:
1. semtrain.p- training data (SemEval 2014)
2. semtest.p- testing date (SemEval 2014)
3. stsallrmf.p- all STS data.

Scripts: (in examples folder)
1. example1.py : Load trained model to predict sentence similarity on a scale of 1.0-5.0
2. example2.py : Load trained model and check Pearson, Spearman and MSE.
3. example3.py : Train the model (takes a long time to compile gradients)
4. examples.ipynb : explanation of the MaLSTM code (iPython notebook)


Mueller, J and Thyagarajan, A.  Siamese Recurrent Architectures for Learning Sentence Similarity.  Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016).
 http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195