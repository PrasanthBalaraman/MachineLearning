from __future__ import print_function 
import cntk as C 
import os 
import sys 
import numpy as np 
import requests

source_data = {
  'train': { 'file': 'PCD_tagger_shuffled_train_v6.ctf', 'location': 0 },
  'test': { 'file': 'PCD_tagger_shuffled_test_v6.ctf', 'location': 0 },
  'sentence': { 'file': 'sentence_v6.wl', 'location': 0 },
  'tag': { 'file': 'tag_v6.wl', 'location': 0 }
}

seq_list = []
with open("PCD_tagger_shuffled_test_v6.txt", "r") as file:
    for line in file.readlines():
        seq_list.append(((line.split("\t")[0]).strip(), (line.split("\t")[1]).strip()))

vocab_size = 2556
num_intents = 3

input_dim = vocab_size
embed_dim = 100
hidden_dim = 150
num_classes = num_intents

dropout_rate = 0.2

x = C.sequence.input_variable(input_dim)
y = C.input_variable(num_classes)

def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        query = C.io.StreamDef(field="S0", shape=vocab_size, is_sparse=True),
        intent = C.io.StreamDef(field="S1", shape=num_intents, is_sparse=True),
    )))

def OneWordLookahead():
    x = C.placeholder()
    apply_x = C.splice(x, C.sequence.future_value(x))
    return apply_x

def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x 

def create_model():
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(embed_dim, name="embed"),
            #OneWordLookahead(),
            # Please use batch normalization in GPU as it is not implemented in CPU yet by Microsoft
            #C.layers.BatchNormalization(),
            C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False),
            #C.layers.BatchNormalization(),
            C.layers.sequence.last,
            C.layers.Dropout(dropout_rate=dropout_rate, name='DropoutLayer'),
            C.layers.Dense(num_classes, name="classify")
        ])

z = create_model()
reader = create_reader(source_data['train']['file'], is_training=True)

def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)

def train(reader, model_func, max_epochs=200):
    
    model = model_func(x)
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 500
    minibatch_size = 64
    
    lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(700)
    
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True,
                     l2_regularization_weight=0.001)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: reader.streams.query,
                y: reader.streams.intent
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()
    
    # Save the trainer after training 
    trainer.save_checkpoint("sequence_classifier.dnn")

    reader_test = create_reader(source_data['test']['file'], is_training=False)

    test_input_map = {
        x: reader.streams.query,
        y: reader.streams.intent
    }
    
    # Test data for trained model
    test_minibatch_size = 1
    num_samples = 96
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0

    for i in range(num_minibatches_to_test):
        data = reader_test.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("\n")
    print("Average test error: {0:.2f}%".format(test_result*100/ num_minibatches_to_test))
    print("\n")

output_filename = "model_predicted_output.csv"
output_file = open(output_filename, "w+")

def vector_to_sentence(seq,tag, model):
    sentence_wl = [line.rstrip('\n') for line in open(source_data['sentence']['file'], encoding="utf-8")]
    tag_wl = [line.rstrip('\n') for line in open(source_data['tag']['file'], encoding="utf-8")]
    sentence_dict = {sentence_wl[i]:i for i in range(len(sentence_wl))}
    tag_dict = {tag_wl[i]:i for i in range(len(tag_wl))}

    test_seq = seq
    w = [sentence_dict[w] for w in test_seq.split()]
    #print(w)
    onehot = np.zeros([len(w),len(sentence_dict)], np.float32)
    for t in range(len(w)):
        onehot[t,w[t]] = 1

    pred = model(x).eval({x:[onehot]})[0]
    best = np.argmax(pred)
    #print(str(pred))
    #print("Sentence: " + test_seq + "\n" + "Tag: " + tag_wl[best] + ", " +tag+ "\n")
    output_file.write(test_seq + "," + tag_wl[best] + "," + tag +","+str(pred)+"\n")

def do_train_test():
    global z
    z = create_model()
    reader = create_reader(source_data['train']['file'], is_training=True)
    train(reader, z, max_epochs=100)
    for (seq, tag) in seq_list:
        vector_to_sentence(seq, tag, z)

do_train_test()

z.save(os.path.join("C:\Deep Learning\CNTK\model", "saved_intent_classifer" + ".dnn"))
