import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import Util as Data
import re

util = Data.Util()
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = util.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = util.get_largest_num_of_words_in_answer()
special_chars = ["'", "/", ")", "(", "/", "'", "[", "{", "]", "}", "#", "$", "%",
                              "^", "&", "*", "-", "_", "+", "=", ".", "\"", ",", ":", ";"]
largest_num_of_words_any_paragraph = 200
get_largest_num_of_words_in_question = 20

def vectorise_paragraph(par):
    paragraphs_sentences = np.zeros((largest_num_of_words_any_paragraph, 200))
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', par)
    v = 0
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            characters = list(word)
            if len(characters) > 0:
                if characters[0] in special_chars:
                    glove_embedding = util.get_glove_embedding(characters[0])
                    paragraphs_sentences[v] = glove_embedding
                    v = v + 1
                    if v >= largest_num_of_words_any_paragraph:
                        break
                    word = word[1:]
                if characters[len(characters) - 1] in special_chars:
                    word = word[:-1]
                word = word.lower()
                if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                    apostrophe_word = word.split("'")
                    glove_embedding = util.get_glove_embedding(apostrophe_word[0])
                    paragraphs_sentences[v] = glove_embedding
                    v = v + 1
                    if v >=largest_num_of_words_any_paragraph:
                        break
                    glove_embedding = util.get_glove_embedding("'" + apostrophe_word[1])
                    paragraphs_sentences[v] = glove_embedding
                    v = v + 1
                    if v >= largest_num_of_words_any_paragraph:
                        break
                else:
                    glove_embedding = util.get_glove_embedding(word)
                    paragraphs_sentences[v] = glove_embedding
                    v = v + 1
                    if v >= largest_num_of_words_any_paragraph:
                        break
                if characters[len(characters) - 1] in special_chars:
                    glove_embedding = util.get_glove_embedding(characters[len(characters) - 1])
                    paragraphs_sentences[v] = glove_embedding
        v = v + 1
        if v >= largest_num_of_words_any_paragraph:
            break
    return paragraphs_sentences


def vectorise_question(ques):
    questions_words = np.zeros((38, 200))
    words = ques.split(' ')
    v = 0
    for word in words:
        characters = list(word)
        if len(characters) > 0:
            if characters[0] in special_chars:
                glove_embedding = util.get_glove_embedding(characters[0])
                questions_words[v] = glove_embedding
                v = v + 1
                word = word[1:]
            if characters[len(characters) - 1] in special_chars:
                word = word[:-1]
            word = word.lower()
            if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                apostrophe_word = word.split("'")
                glove_embedding = util.get_glove_embedding(apostrophe_word[0])
                questions_words[v] = glove_embedding
                v = v + 1
                glove_embedding = util.get_glove_embedding("'" + apostrophe_word[1])
                questions_words[v] = glove_embedding
                v = v + 1
            else:
                glove_embedding = util.get_glove_embedding(word)
                questions_words[v] = glove_embedding
                v = v + 1
            if characters[len(characters) - 1] in special_chars:
                glove_embedding = util.get_glove_embedding(characters[len(characters) - 1])
                questions_words[v] = glove_embedding
                v = v + 1
    return questions_words

def get_answer(par, ques):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["tag"], "model")
        graph = tf.get_default_graph()
        question = graph.get_tensor_by_name("question:0")
        text = graph.get_tensor_by_name("text:0")
        answer_softmax = graph.get_tensor_by_name("train/model/encoder-decoder/answer:0")
        paragraphs_sentences = vectorise_paragraph(par)
        questions_words = vectorise_question(ques)

        feed_dict = {question: questions_words, text: paragraphs_sentences}
        classification = sess.run(answer_softmax, feed_dict)
        classif_words = util.get_words(classification)
        a = ""
        for word in classif_words:
            a = a + " " + word
        a = a + "."
        return a[1:].capitalize()
