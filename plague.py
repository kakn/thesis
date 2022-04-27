"""
Bachelor Project

10/02/22

* Reddit data
* ssh kakn@hpc.itu.dk
* Write to ssh - tight deadline and cc christian

= Indent error lol
= Was only using :400 pairs...
= Wasn't utilizing cornell conversations, q&as were hit or miss before
= Will def need cluster

+ Make function names better
+ Save history!!!!!!!

? Prevec word split or not?
? Look into dimensionality

! What happens if i put a sentence from the training set
! I need to have a research question that I'm answering and use some method to test it, get some results and interpret them
! Investigate tradeoff between training time and quality , data science and quality,
! Data augmentation - is there a way to generate more data for varied domains artifically - is there a way to extract data from additional corpus - in order to expand on domains from data that isn't necessarily conversation
! As open domain as possible - how?
! Look into research papers that maximize domain coverage of chatbots
! Turing test
! Self-attention as a Non-recurrent neural network
! Pycharm
! Deep Learning Ian Goodfellow - Recurrent and Recursive Neural Nets

!! 167709 lines, should I experiment with small samples?
!! Training will take fucking ages, increase batch size and use smaller random sample
!! Epochs linear, training size incredibly exponential, batch_size middle ground. Not sure about dimensionality
!! Batch size: for each gradient update it takes a bunch of training examples (batches), then average the gradient of that
!! A few different models with different data splits - run with different hyperparameters and present domain coverage
!! Genre classification of these movies - would allow me to split up training set in different categories - domain coverage
!! Similarity between movie quotes - use bleu or perplexity to feed the model sentences it already knows and see how far away it is from original answer
!! Or get friends to rate the bot - disfluent, semi coherent or coherent
!! Cornell movie genre - 5 of them
!! Which genres
!! Bleu and perplexity
!! Comedy
!! Very important to use different test data than train
!! The number of epochs balances with the training size in the number of gradient updates you get ¨
!! Have 2 models for each genre
"""

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

def reader():
    #nard = open("movie_lines.txt", encoding = "ISO-8859-1")
    # Importing the dataset
    lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    # Creating a dictionary that maps each line and its id
    id2line = {}
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
    return pairs

# Pre-vectorization step

def prevec(pairs):
    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()

    for line in pairs[:2000]:
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
    #Dimensionality
    dimensionality = 256
    #The batch size and number of epochs
    batch_size = 100
    epochs = 1000
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
    return batch_size, epochs, encoder_inputs, encoder_states, decoder_outputs, decoder_inputs

def training(encoder_inputs, decoder_inputs, decoder_outputs, encoder_input_data, decoder_input_data, decoder_target_data, epochs, batch_size):
    #Model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #Compiling
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
    #Training
    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
    training_model.save('training_model.h5')

def modellen(decoder_inputs, decoder_lstm):
    global decoder_model
    global encoder_model

    training_model = load_model('training_model.h5')
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

def trainor():
    questions, answers = reader()
    pairs = preproc(questions, answers)
    input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens = prevec(pairs)
    input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_encoder_seq_length, max_decoder_seq_length, encoder_input_data, decoder_input_data, decoder_target_data = vectorizer(input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens)
    batch_size, epochs, encoder_inputs, encoder_states, decoder_outputs, decoder_inputs = setup(num_encoder_tokens, num_decoder_tokens)
    #training(encoder_inputs, decoder_inputs, decoder_outputs, encoder_input_data, decoder_input_data, decoder_target_data, epochs, batch_size)

def main():
    trainor()
    modellen(decoder_inputs, decoder_lstm)
    Aeris()
main()
