import numpy as np
import sys,logging,re,os
import tarfile

from gensim.models.keyedvectors import KeyedVectors as kv
from progress.bar import Bar

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def filter_words(words,emb):
   if words is None:
       return
   return [word for word in words if word in emb.vocab]

def get_utterance_embedding_w2v(emb,utterance,emb_size=0):

    if emb_size == 0:
        if type(emb) == type({}):
            emb_size = len(emb)
        elif type(emb) is np.ndarray:
            emb_size = emb.shape[1]
        else:
            logging.error('Type of embedding unknown')
            sys.exit()

    if utterance == 'silence':
        input('silence')
        return
    else:
        utterance = filter_words(utterance,emb)

        #for unseen words return a randomly initialized vector
        if utterance == []:
           return 2*np.random.random((emb.vector_size,))-1

        wordVectorsMatrix = np.empty((len(utterance),emb.vector_size))
        for w,word in enumerate(utterance):
           wordVectorsMatrix[w,:] = emb[word]

        return np.mean(wordVectorsMatrix,axis=0)


def get_utterance_embedding(emb,utterance,emb_size=0,wrd2idx=None):

    if not wrd2idx:
        logging.error('Please provide word to index dictionary')

    utterance = pre_process_utterance(utterance.lower())

    emb_matrix = np.zeros((len(utterance.split()), emb_size))

    for w,wrd in enumerate(utterance.split()):
        if wrd.lower() in wrd2idx:
            emb_matrix[w] = emb[wrd2idx[wrd]]
        elif wrd.lower() == 'let\'s':
            # exception for the case of let's
            emb_matrix[w] = (emb[wrd2idx['let']] + emb[wrd2idx['us']]) / 2
        elif wrd.lower() == '<silence>' or wrd == '<SIL>' or wrd == '<sil>':
            try:
                emb_matrix[w] = emb[wrd2idx['<SIL>']]
            except:
                print(emb.shape)
                input(wrd2idx['<SIL>'])
        else:
            if 'OOV' in wrd2idx:
                emb_matrix[w] = emb['OOV']
            else:
                emb_matrix[w] = 2*np.random.random((emb.shape[1],))-1
            logging.debug('{} not found in the glove model'.format(wrd))

    tmp_mat = np.mean(emb_matrix,axis=0)
    for v in tmp_mat:
        if v > 10:
            input(v)

    return np.mean(emb_matrix,axis=0)


def pre_process_utterance(utt):
    '''
    removes abreviations and underscores from sentence
    for word embedding purposes

    :param utt:
    :return:
    '''

    pre_pro_utt = re.sub('_', ' ', utt)
    pre_pro_utt = re.sub("i'd",'i would',pre_pro_utt)
    pre_pro_utt = re.sub("don't",'do not',pre_pro_utt)
    pre_pro_utt = re.sub("it's",'it is',pre_pro_utt)

    return pre_pro_utt


def loading_gensim_embeddings(embeddings_file):

    logging.info('loading embeddings from {}'.format(embeddings_file))
    #emb = w2v.load_word2vec_format(fileName.strip(),binary=True)
    emb = kv.load_word2vec_format(embeddings_file,binary=True)

    _eos_ = 2 * np.random.random((emb.vector_size,)) - 1
    _sil_ = 2 * np.random.random((emb.vector_size,)) - 1
    emb.wv.add('<EOS>',_eos_)
    emb.wv.add('<SIL>',_sil_)

    return emb


def load_glove_vectors_w2v(glove_file_location,dimension,corpus_size):

    if glove_file_location.endswith('.npz') or\
        glove_file_location.endswith('.npy'):

        logging.error('Please use load_glove_vector instead')
        sys.exit()

    glove_dir = os.sep.join(glove_file_location.split(os.sep)[:-1])

    model = np.empty((dimension,))

    w2v_file = os.path.join(glove_dir, 'glove.{}d.{}.w2v.bin'.format(dimension, corpus_size))

    if not os.path.isfile(w2v_file):
        _ = glove2word2vec(glove_file_location, w2v_file)

        model = KeyedVectors.load_word2vec_format(w2v_file,binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(w2v_file)
    _eos_ = 2 * np.random.random((dimension,)) - 1
    _sil_ = 2 * np.random.random((dimension,)) - 1
    model.wv.add('<EOS>',_eos_)
    model.wv.add('<SIL>',_sil_)

    logging.info('{} words loaded'.format(len(model.wv.vocab)))
    return model,_

def load_glove_vectors(glove_file_location,glove_dim,corpus_size=0):
    if glove_file_location.endswith('.txt') or \
            glove_file_location.endswith('.npz') or \
            glove_file_location.endswith('.gz'):
        if not os.path.isfile(glove_file_location):
            logging.error('Couldn\'t find file {}'.format(glove_file_location))
            sys.exit()
        np_golve_file = '{}.npz'.format(glove_file_location.rsplit('.',1)[0])
        if not os.path.isfile(np_golve_file):
            if glove_file_location.endswith('txt'):
                word2idx = []
                embedding_matrix = []
                logging.info('Loading Glove Vector from text file')
                with open(glove_file_location, 'r') as f:
                    first_line = f.readline()
                    data = [x.strip().lower() for x in first_line.split()]
                    word = data[0]
                    word2idx.append(word)
                    embedding_matrix.append(np.asarray(data[1:glove_dim + 1 ], dtype='float32'))
                    bar = Bar('Loading Word Emebeddings:')
                    for idx, line in enumerate(f):
                        bar.next()
                        try:
                            data = [x.strip().lower() for x in line.split()]
                            word2idx.append(data[0])
                            embedding_matrix.append(np.asarray(data[1: glove_dim + 1], dtype='float32'))
                        except Exception as e:
                            print('Exception in loading glove word embeddings {}'.format(e))
                            continue

            elif glove_file_location.endswith('gz'):
                embedding_matrix = []
                word2idx = []
                logging.info('Unpacking {}'.format(glove_file_location))
                tar = tarfile.open(glove_file_location,"r:gz")
                logging.info('Done')
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    first_line = f.readline()
                    input(first_line)
                    embedding_dim = len([x.strip() for x in first_line.strip()]) - 1
                    data = [x.strip().lower() for x in first_line.split()]
                    word = data[0]
                    word2idx.append(word)
                    embedding_matrix.append(np.asarray(data[1:embedding_dim + 1], dtype='float32'))
                    bar = Bar()
                    for idx, line in enumerate(f):
                        bar.next()
                        try:
                            data = [x.strip().lower() for x in line.split()]
                            word = data[0]
                            word2idx.append(word)
                            embedding_matrix.append(np.asarray(data[1:embedding_dim + 1],dtype='float32'))
                        except Exception as e:
                            logging.error('Exception occurred in loading embeddings:\n{}'.format(e))
                            continue
                #embedding_matrix = np.asarray(embedding_matrix)
            else:
                logging.error('File extension not supported')
                sys.exit()
            # adding fixed embeddings to the model for unseen words
            for w,wrd in enumerate(['<EOS>','<SIL>']):
                embedding_matrix = np.vstack((embedding_matrix,2 * np.random.random((glove_dim,)) - 1)) #randomly assigning a value to the end of sentence
                word2idx.append(wrd)
            logging.info('Saving embeddings file as numpy file')
            np.savez_compressed(open(np_golve_file,'wb'),embedding=embedding_matrix,word2idx=word2idx)
            logging.info('{} words loaded'.format(len(word2idx)))
            return embedding_matrix,word2idx
        else:
            glove_file_location = np_golve_file
    else:
        logging.error('GloVe file extension not supported')
        sys.exit()
    logging.info('Loading word embeddings from binary file {}...'.format(glove_file_location))
    loaded = np.load(glove_file_location)
    embedding_matrix = loaded['embedding']
    logging.debug('Embedding shape {}'.format(embedding_matrix.shape))
    sil_vec = 2*np.random.random((glove_dim,)) - 1
    embedding_matrix = np.vstack((embedding_matrix,sil_vec)) #randomly assigning the same value for <SILENCE>
    word2idx = dict()
    for i, word in enumerate(loaded['word2idx']):
        word2idx[word] = i
    if '<SIL>' not in word2idx:
        word2idx['<SIL>'] = i+1
    del loaded
    logging.info('{} words loaded'.format(len(word2idx)))
    return embedding_matrix, word2idx
