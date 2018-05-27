import os
import warnings
import sys
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np;
import re
from collections import OrderedDict

from keras.models import Model;
from keras.models import load_model;
#from keras.optimizers import rmsprop,adam,adagrad,SGD;
#from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau, CSVLogger;
from keras.preprocessing.text import one_hot,Tokenizer;
#from keras.layers import Input,Dense,merge,Dropout,BatchNormalization,Activation,Conv1D;



maxlen=205;
model_path = 'models/' +'conv1d_54'#'conv1d_30'


def input_2_vec(input):
    source=input[0];target=input[1];
    source_vec=np.zeros((1,maxlen+1),dtype=np.uint16);
    target_vec=np.zeros((1,maxlen),dtype=np.uint16);
    for i,ele1 in enumerate(source):

        if ele1 not in french_index:
            raise Exception('Символ "%s" не входит в словарь модели!'%ele1)
        source_vec[0,i]=french_index[ele1];
    for j,ele2 in enumerate(target):
        target_vec[0,j]=eng_index[ele2];
    return (source_vec,target_vec);

def T(model, sentence):
    vec=input_2_vec(sentence);
    source_vec=vec[0];target_vec=vec[1];
    t0=np.zeros((1,1,500),dtype=np.uint8);
    predict=model.predict([source_vec,target_vec,t0])[0];
    predict_max=np.argmax(predict,axis=-1);
    answers=[index_eng[j+1] for j in predict_max];
    a="".join(answers);
    return a;

#plot_model(model, to_file="model.png", show_shapes=True)
def translate(model, french_sentence):
    french_sentence = preproc_test(french_sentence)
    process_english_sentence="";
    length=0;
    while 1:
        predicted_english_sentence=T(model, [french_sentence,process_english_sentence]);
        process_english_sentence=predicted_english_sentence[:length+1];length+=1;
        if process_english_sentence[-1]=="\n":break;
        if length>=maxlen+1:break;
        #if length%10==0:print("{} completed".format(str(length*maxlen**-1)));
    return process_english_sentence.strip();

def preproc_test(line):
    
    replace_dict = OrderedDict({
                                    '<...>': '',
                                    '«': '', '»': '',
                                    '...': '',
                                    '!.': '.',
                                    '!': '.',
                                    '?.': '.',
                                    '?': '.',     
                                    ';': ',', #!!!
    })
    
    for k in replace_dict.keys():
        line = line.replace(k, replace_dict[k])
        
    
    line = re.sub(r"\[\d+\]", "", line)
    line = re.sub(r"\(\d+\)", "", line)
    line = re.sub(r" \.", ".", line)

    replace_dict_2 = OrderedDict({
        '.,': '.',
        '—': '',
        '“': '',
        '”': '',
        '’': '',
        '˝': '',
        '"': '',
        '´': '',
        '(': '', ')': '',
        '<': '', '>': '',
        '-': '', 
        '„': '',
        '̀': '',
        '́': '',
        '*': '',
        '…': '',  
        "'": '',
        '{': '',
        '}': '',
        '[': '',
        ']': '',
                                
    })
    
    for k in replace_dict_2.keys():
        line = line.replace(k, replace_dict_2[k])
                

    line = line.replace('\t', ' ')
    line = line.replace(',', '') #!!
    #line = line.replace('.', '') #!!!
    line = line.replace(':', '')
    
    line = line.replace('.', '')
    
    line = line.lower() #!!!
    
    return line


eng_index, french_index, index_eng, index_french = ({' ': 1,
      'о': 2,
      'и': 3,
      'е': 4,
      'а': 5,
      'с': 6,
      'т': 7,
      'н': 8,
      'в': 9,
      'р': 10,
      'м': 11,
      'л': 12,
      'д': 13,
      'у': 14,
      'п': 15,
      'к': 16,
      'я': 17,
      'ъ': 18,
      'г': 19,
      'ѣ': 20,
      'б': 21,
      'ь': 22,
      'ж': 23,
      'ы': 24,
      'з': 25,
      '\n': 26,
      'ш': 27,
      'х': 28,
      'ч': 29,
      'ю': 30,
      'й': 31,
      'ц': 32,
      'щ': 33,
      'ф': 34,
      '0': 35,
      '6': 36,
      '1': 37,
      '4': 38,
      '5': 39,
      '3': 40,
      '7': 41,
      '2': 42,
      '8': 43,
      '9': 44,
      'ѝ': 45,
      'э': 46,
      'і': 47,
      'ё': 48,
      'à': 49,
      'c': 50,
      'i': 51},
     {' ': 1,
      'о': 2,
      'и': 3,
      'е': 4,
      'а': 5,
      'с': 6,
      'т': 7,
      'н': 8,
      'в': 9,
      'л': 10,
      'р': 11,
      'м': 12,
      'д': 13,
      'к': 14,
      'п': 15,
      'у': 16,
      'г': 17,
      'я': 18,
      'б': 19,
      'ы': 20,
      'ь': 21,
      'з': 22,
      'ч': 23,
      'й': 24,
      'ж': 25,
      'х': 26,
      'ш': 27,
      'ю': 28,
      'ц': 29,
      'щ': 30,
      'э': 31,
      'ф': 32,
      '6': 33,
      '4': 34,
      '7': 35,
      '5': 36,
      'ъ': 37,
      '1': 38,
      '0': 39,
      '3': 40,
      '2': 41,
      '9': 42,
      '8': 43,
      'ο': 44,
      'a': 45,
      'κ': 46,
      'h': 47,
      'e': 48,
      'y': 49,
      'ё': 50,
      'c': 51,
      'é': 52,
      't': 53,
      'o': 54},
     {1: ' ',
      2: 'о',
      3: 'и',
      4: 'е',
      5: 'а',
      6: 'с',
      7: 'т',
      8: 'н',
      9: 'в',
      10: 'р',
      11: 'м',
      12: 'л',
      13: 'д',
      14: 'у',
      15: 'п',
      16: 'к',
      17: 'я',
      18: 'ъ',
      19: 'г',
      20: 'ѣ',
      21: 'б',
      22: 'ь',
      23: 'ж',
      24: 'ы',
      25: 'з',
      26: '\n',
      27: 'ш',
      28: 'х',
      29: 'ч',
      30: 'ю',
      31: 'й',
      32: 'ц',
      33: 'щ',
      34: 'ф',
      35: '0',
      36: '6',
      37: '1',
      38: '4',
      39: '5',
      40: '3',
      41: '7',
      42: '2',
      43: '8',
      44: '9',
      45: 'ѝ',
      46: 'э',
      47: 'і',
      48: 'ё',
      49: 'à',
      50: 'c',
      51: 'i'},
     {1: ' ',
      2: 'о',
      3: 'и',
      4: 'е',
      5: 'а',
      6: 'с',
      7: 'т',
      8: 'н',
      9: 'в',
      10: 'л',
      11: 'р',
      12: 'м',
      13: 'д',
      14: 'к',
      15: 'п',
      16: 'у',
      17: 'г',
      18: 'я',
      19: 'б',
      20: 'ы',
      21: 'ь',
      22: 'з',
      23: 'ч',
      24: 'й',
      25: 'ж',
      26: 'х',
      27: 'ш',
      28: 'ю',
      29: 'ц',
      30: 'щ',
      31: 'э',
      32: 'ф',
      33: '6',
      34: '4',
      35: '7',
      36: '5',
      37: 'ъ',
      38: '1',
      39: '0',
      40: '3',
      41: '2',
      42: '9',
      43: '8',
      44: 'ο',
      45: 'a',
      46: 'κ',
      47: 'h',
      48: 'e',
      49: 'y',
      50: 'ё',
      51: 'c',
      52: 'é',
      53: 't',
      54: 'o'})


def main():

    args = sys.argv
    if len(args) == 1:
        print("Отсутствует строка для перевода")
    elif len(args) > 2:
        print("Слишком много входных аргументов")
    else:
        s = args[1]
        #s = 'Здравствуй, как жизнь?'
        model=load_model(model_path)
        t = translate(model, french_sentence=s)
        print(t)



if __name__ == "__main__":
    main()
