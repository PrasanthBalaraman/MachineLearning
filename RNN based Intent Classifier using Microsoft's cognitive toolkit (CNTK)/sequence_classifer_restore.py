from __future__ import print_function 
import cntk as C 
import os 
import sys 
import numpy as np 
import requests

vocab_size = 2556
input_dim = vocab_size
x = C.sequence.input_variable(input_dim)

source_data = {
  'sentence': { 'file': 'sentence_v6.wl', 'location': 0 },
  'tag': { 'file': 'tag_v6.wl', 'location': 0 }
}

z = C.load_model(os.path.join("C:\Deep Learning\CNTK\model", "saved_intent_classifer" + ".dnn"))
out = C.softmax(z)

def vector_to_sentence(seq, model):
    sentence_wl = [line.rstrip('\n') for line in open(source_data['sentence']['file'], encoding="utf-8")]
    tag_wl = [line.rstrip('\n') for line in open(source_data['tag']['file'], encoding="utf-8")]
    sentence_dict = {sentence_wl[i]:i for i in range(len(sentence_wl))}
    tag_dict = {tag_wl[i]:i for i in range(len(tag_wl))}

    test_seq = seq
    w = []
    for word in test_seq.split():
        try:
            w.append(sentence_dict[word])
        except KeyError:
            w.append(2419)
    #print(w)

    #print(type(w))
    onehot = np.zeros([len(w),len(sentence_dict)], np.float32)
    for t in range(len(w)):
        onehot[t,w[t]] = 1

    pred = model(x).eval({x:[onehot]})[0]
    best = np.argmax(pred)
    #print(str(pred))
    print("\nInput Sentence:\t" + test_seq + "\n" + "Tag:\t\t" + tag_wl[best] + "\n")
    output_file.write(test_seq + "," + tag_wl[best] + "," +str(pred)+"\n")

# # predicting one sentence at a time
# vector_to_sentence("", z)


# Use below to iteratively predict for input as CSV and save the output in CSV
seq_list = []
with open(".\Shravan\\test_file.txt", "r", encoding="utf-8") as file:
    for line in file.readlines():
        seq_list.append(((line.split("\t")[0]).strip(), (line.split("\t")[1]).strip()))

output_filename = "model_infered_output.csv"
output_file = open(output_filename, "w+", encoding="utf-8")
output_file.write( "Input Sentence, Predicted Tag, Probailities [customer_impact log_snippet diagnostic_info]\n")

for (seq, tag) in seq_list:
        vector_to_sentence(seq, z)