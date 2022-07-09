import re
import string
import time
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import CSVLogger
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def testdata(gen, big=False):
    lines = open('/Users/kasper/bojack/sem6/thesis/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    conversations = open('/Users/kasper/bojack/sem6/thesis/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    metadata = open('/Users/kasper/bojack/sem6/thesis/movie_titles_metadata.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    swtor = ["m3", "m5", "m34", "m125", "m337", "m489", "m529", "m531", "m328", "m253", "m433"]
    genres = ["'thriller'", "'comedy'", "'horror'", "'drama'"]
    movid = []
    for line in metadata[:-1]:
        _line = line.split(' +++$+++ ')
        genre = _line[-1].split(',')
        if len(genre) == 1:
            p = genre[0].strip("").strip('[').strip(']')
            if gen == p:
                movid.append(_line[0])
            if gen == "'sci-fi'":
                for s in swtor:
                    movid.append(s)
            if gen == "'combo'":
                if p in genres:
                    movid.append(_line[0])
                for s in swtor:
                    movid.append(s)
        if big == True:
            if gen == "'horror'" or gen == "'combo'":
                if len(genre) == 2:
                    for i in range(2):
                        p = genre[i].strip("").strip('[').strip(']')
                        if p == "'horror'":
                            if _line[0] not in movid:
                                movid.append(_line[0])
    id2line = {}
    pines = []
    ponversations = []
    for line in lines[:-1]:
        _line = line.split(' +++$+++ ')
        if _line[2] in movid:
            pines.append(line)
    for conv in conversations[:-1]:
        _conv = conv.split(' +++$+++ ')
        if _conv[2] in movid:
            ponversations.append(conv)
    lines = pines
    conversations = ponversations
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    # Creating a list of all conversations
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))
    # Separating questions and answers
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i+1]])
    pairs = preproc(questions, answers)
    if big == True:
        return pairs[2000:]
    else:
        return pairs[1000:]

def reader():
    #nard = open("movie_lines.txt", encoding = "ISO-8859-1")
    # Importing the dataset
    lines = open('/Users/kasper/bojack/sem6/thesis/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    conversations = open('/Users/kasper/bojack/sem6/thesis/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    metadata = open('/Users/kasper/bojack/sem6/thesis/movie_titles_metadata.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    # Creating a dictionary that maps each line and its id
    movid = ["m3", "m5", "m34", "m125", "m337", "m489", "m529", "m531", "m328", "m253", "m433"]
    genres = ["'thriller'", "'comedy'", "'horror'", "'drama'"]
    for line in metadata[:-1]:
        _line = line.split(' +++$+++ ')
        genre = _line[-1].split(',')
        if len(genre) == 1:
            p = genre[0].strip("").strip('[').strip(']')
            if p in genres:
                movid.append(_line[0])
        # if len(genre) == 2:
        #     for i in range(2):
        #         p = genre[i].strip("").strip('[').strip(']')
        #         if p == "'horror'":
        #             if _line[0] not in movid:
        #                 movid.append(_line[0])
    id2line = {}
    pines = []
    ponversations = []
    for line in lines[:-1]:
        _line = line.split(' +++$+++ ')
        if _line[2] in movid:
            pines.append(line)
    for conv in conversations[:-1]:
        _conv = conv.split(' +++$+++ ')
        if _conv[2] in movid:
            ponversations.append(conv)
    lines = pines
    conversations = ponversations
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    # Creating a list of all conversations
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))
    # Separating questions and answers
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i+1]])
    return questions, answers

# Cleaning texts

def clean_text(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# Correcting cleaned texts for output
def botcleaner(sentence):
    caps = sentence[0].upper()
    newsent = caps + sentence[1:]
    return newsent

def punctrem(sent):
    new_string = sent.translate(str.maketrans('', '', string.punctuation))
    return new_string

# Necessary preprocessing

def preproc(questions, answers):
    # Cleaning the questions
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))
    # Cleaning the answers
    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # Filtering out the questions and answers that are too short or too long
    short_questions = []
    short_answers = []
    i = 0
    for question in clean_questions:
        if 2 <= len(question.split()) <= 25:
            short_questions.append(question)
            short_answers.append(clean_answers[i])
        i += 1
    clean_questions = []
    clean_answers = []
    i = 0
    for answer in short_answers:
        if 2 <= len(answer.split()) <= 25:
            clean_answers.append(answer)
            clean_questions.append(short_questions[i])
        i += 1

    # Combining Q&A's for next step
    pairs = []
    for i in range(len(clean_questions)):
        pairs.append([clean_questions[i], clean_answers[i]])
    random.seed(42)
    random.shuffle(pairs)
    #print(len(pairs))
    return pairs

# Pre-vectorization step

def prevec(pairs):
    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()

    for line in pairs[:1000]:
      input_doc, target_doc = line[0], line[1]
      # Appending each input sentence to input_docs
      input_docs.append(input_doc)
      # Splitting words from punctuation
      target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
      # Redefine target_doc below and append it to target_docs
      target_doc = '<START> ' + target_doc + ' <END>'
      target_docs.append(target_doc)

      # Now we split up each sentence into words and add each unique word to our vocabulary set
      for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
          input_tokens.add(token)
      for token in target_doc.split():
        if token not in target_tokens:
          target_tokens.add(token)

    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))
    global num_encoder_tokens
    global num_decoder_tokens
    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)

    return input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens

def vectorizer(input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens):
    global input_features_dict
    global reverse_input_features_dict
    global target_features_dict
    global reverse_target_features_dict
    global max_encoder_seq_length
    global max_decoder_seq_length

    input_features_dict = dict(
        [(token, i) for i, token in enumerate(input_tokens)])
    target_features_dict = dict(
        [(token, i) for i, token in enumerate(target_tokens)])

    reverse_input_features_dict = dict(
        (i, token) for token, i in input_features_dict.items())
    reverse_target_features_dict = dict(
        (i, token) for token, i in target_features_dict.items())

    max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
    max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

    encoder_input_data = np.zeros(
        (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
        for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
            #Assign 1. for the current line, timestep, & word in encoder_input_data
            encoder_input_data[line, timestep, input_features_dict[token]] = 1.

        for timestep, token in enumerate(target_doc.split()):
            decoder_input_data[line, timestep, target_features_dict[token]] = 1.
            if timestep > 0:
                decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

    return input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_encoder_seq_length, max_decoder_seq_length, encoder_input_data, decoder_input_data, decoder_target_data

def setup(num_encoder_tokens, num_decoder_tokens):
    global decoder_inputs
    global decoder_lstm
    global decoder_dense
    global training_model
    #Dimensionality
    dimensionality = 256
    #The batch size and number of epochs
    batch_size = 32
    epochs = 1200
    #Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_lstm = LSTM(dimensionality, return_state=True)
    encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
    encoder_states = [state_hidden, state_cell]
    #Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    #Model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #Compiling
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
    return batch_size, epochs, encoder_inputs, encoder_states, decoder_outputs, decoder_inputs, training_model

nowtrain = False

def training(training_model, encoder_input_data, decoder_input_data, decoder_target_data, epochs, batch_size):
    #Training
    start_time = time.time()
    nowtrain = True
    csv_logger = CSVLogger("s-combo_model_history_log.csv", append=True)
    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2, callbacks=[csv_logger])
    training_model.save_weights('s-combo.h5')
    print("--- %s seconds ---" % (time.time() - start_time))

def modellen(training_model, decoder_inputs, decoder_lstm):
    global decoder_model
    global encoder_model

    if nowtrain == False:
        training_model.load_weights('s-combo.h5')
        print("working!")

    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    latent_dim = 256
    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_response(test_input):
    #Getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)
    #Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    #Setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['<START>']] = 1.
    #A variable to store our response word by word
    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        #Predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        #Choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token
        #Stop if hit max length or found the stop token
        if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        #Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        #Update states
        states_value = [hidden_state, cell_state]
    return decoded_sentence

def string_to_matrix(user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for timestep, token in enumerate(tokens):
        if token in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix

def generate_response(user_input):
    user_input = clean_text(user_input)
    input_matrix = string_to_matrix(user_input)
    # print("Cleaned sentence: '{s}'".format(s=user_input))
    # counter = 0
    # for row in input_matrix:
    #   for r in row:
    #       for i in r:
    #           if i == 1:
    #               counter += 1
    # print("Length of sentence: {l}".format(l=len(user_input.split())))
    # print("Number of recognized tokens: {c}".format(c=counter))
    chatbot_response = decode_response(input_matrix)
    #Remove <START> and <END> tokens from chatbot_response
    chatbot_response = chatbot_response.replace("<START>",'')
    chatbot_response = chatbot_response.replace("<END>",'')
    return chatbot_response

def Aeris():
    #print("Aeris v1.2")
    print("--ready--")
    while True:
        user_reply = input()
        reply = generate_response(user_reply)
        print(reply)

def testra():
    start_time = time.time()
    genres = ["'combo'", "'comedy'", "'drama'", "'horror'", "'thriller'", "'sci-fi'"]
    summary = []
    data = []
    for genre in genres:
        i = 0
        scores = []
        pairs = testdata(genre)
        #print(len(pairs))
        # supposed to be 383! now 231
        for p in pairs[:231]:
            question = punctrem(p[0])
            reference = [punctrem(p[1]).split()]
            candidate = punctrem(generate_response(question))
            smoothie = SmoothingFunction().method4
            score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
            scores.append(score)
        #print(scores)
        data.append(scores)
        summary.append(np.average(scores))
        i += 1
    np.savetxt('bleuscores.csv', data, delimiter=',')
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print(summary)

def trainor():
    questions, answers = reader()
    pairs = preproc(questions, answers)
    input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens = prevec(pairs)
    input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_encoder_seq_length, max_decoder_seq_length, encoder_input_data, decoder_input_data, decoder_target_data = vectorizer(input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens)
    batch_size, epochs, encoder_inputs, encoder_states, decoder_outputs, decoder_inputs, training_model = setup(num_encoder_tokens, num_decoder_tokens)
    #training(training_model, encoder_input_data, decoder_input_data, decoder_target_data, epochs, batch_size)

def main():
    trainor()
    modellen(training_model, decoder_inputs, decoder_lstm)
    Aeris()
    #testra()
main()
