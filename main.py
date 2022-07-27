from cmath import nan
from flask import Flask, request 
from flask_restful import Api
import json 
import numpy as np
from tensorflow import keras
from keras.models import load_model

## Global Variables 
latent_dim = 150
maxlen=10
encoder_model = nan
decoder_model = nan
reverse_input_char_index = nan 
reverse_target_char_index = nan 
input_token_index = nan
target_token_index = nan
num_decoder_tokens = nan 
max_decoder_seq_length = nan 
max_encoder_seq_length = nan
num_encoder_tokens = nan 

app = Flask(__name__)
api = Api(app)

##Functions
def separeteByTags(inputTexts):
        input_texts_with_tags =[]
        for i in range(len(inputTexts)):
            for j in range(len(inputTexts[i])):
                if(j == len(inputTexts[i]) -1 ):
                    input_text_with_tag = '\t' + inputTexts[i][j] + '\n'
                else:
                    input_text_with_tag = '\t' + inputTexts[i][j] + '\r'

                input_texts_with_tags.append(input_text_with_tag)
        return input_texts_with_tags

def preprocess(inputTexts):
        for i in range(len(inputTexts)):
            global maxlen
            inputTexts[i] = [inputTexts[i][j:j+maxlen] for j in range(0, len(inputTexts[i]),maxlen)]
        inputs_texts_with_tags = separeteByTags(inputTexts)
        global max_encoder_seq_length
        global num_encoder_tokens
        encoder_input_data =  np.zeros((len(inputs_texts_with_tags), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
        for i, (input_text, target_text) in enumerate(zip(inputs_texts_with_tags, inputs_texts_with_tags)):
            for t, char in enumerate(input_text):
                global input_token_index
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        return encoder_input_data
        
def decode_sequence(input_seq):
    repeat = False
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, 2)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
            
        if sampled_char == "\r" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
            repeat = True
            
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence, repeat

### Loads #####
def loadModel():
    model = load_model('s2s')
    loadEncoder(model)
    loadDecoder(model)
    loadConfigurations()
    
def loadEncoder(model):
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs,state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    global encoder_model
    encoder_model = keras.Model(encoder_inputs, encoder_states)
   
def loadDecoder(model):
    decoder_inputs = model.input[1]  # input_2
    global latent_dim
    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    global decoder_model
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    ) 

def loadDicts():
    f = open ('const/input_token_index.json', "r")
    global input_token_index
    input_token_index = json.loads(f.read())
    f.close()

    f = open ('const/target_token_index.json', "r")
    global target_token_index
    target_token_index = json.loads(f.read())
    f.close()
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    global reverse_input_char_index
    global reverse_target_char_index
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def loadConfigurations():
    f = open ('const/num_decoder_tokens.json', "r")
    global num_decoder_tokens
    num_decoder_tokens = json.loads(f.read())
    f.close()

    f = open ('const/num_encoder_tokens.json', "r")
    global num_encoder_tokens
    num_encoder_tokens = json.loads(f.read())
    f.close()

    f = open ('const/max_decoder_seq_length.json', "r")
    global max_decoder_seq_length
    max_decoder_seq_length = json.loads(f.read())
    f.close()

    f = open ('const/max_encoder_seq_length.json', "r")
    global max_encoder_seq_length
    max_encoder_seq_length = json.loads(f.read())
    f.close()

    loadDicts()

@app.route('/',methods=['GET'])
def home():
    return 'Ola esta a fucnionar'

@app.route('/predict/',methods=['POST'])
def predict():
    output_texts =[]
    inputTexts=request.form['data']
    inputTexts=json.loads(inputTexts)
    encoder_input_data = preprocess(inputTexts)
    seq_index = 0
    for i in range(len(inputTexts)):
        repeat = True
        decoded_text =[]
        while repeat:
            input_seq = encoder_input_data[seq_index : seq_index + 1]
            decoded_sentence, repeat = decode_sequence(input_seq)
            seq_index = seq_index + 1   
            decoded_text.append(decoded_sentence[:len(decoded_sentence)-1])
        output_texts.append(''.join(decoded_text))
    return json.dumps(output_texts)
        

if __name__ == '__main__':
    loadModel()
    app.run(debug=True)