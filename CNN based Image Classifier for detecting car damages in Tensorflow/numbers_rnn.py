import tensorflow as tf 
import numpy as np 

digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
                    7: "Seven", 8: "Eight", 9: "Nine", 0: "PAD"}

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

    if rand_seq_len<6:
        rand_odd_ints = np.append(rand_odd_ints, [0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0]*(6-rand_seq_len))
    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

    data = even_sentences+odd_sentences
    seqlens*=2

print(even_sentences[:6])
print(odd_sentences[:6])




