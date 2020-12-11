import collections
import re,os,sys,json
import logging
import pickle
from mmap import mmap

import numpy as np
import h5py
import tensorflow.keras.utils as k_utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import random
from tensorflow.keras import callbacks
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import random

from orca_utils import utils, rig_utils, nlg_utils, keras_utils

DEBUG = False

class ModelInfo():

    def __init__(self,batch_size,concat_depth=False):

        self.max_dial_len = 0
        self.feat_size = 0
        self.layers = 0
        self.epochs = 0
        self.n_classes = 0
        self.config = 'vanilla_hcn'
        self.batch_size = batch_size #default number found in marios script
        self.CD = concat_depth
        self.activation = 'relu'
        self.sep_features = True
        self.replace_context = False
        self.stop_criterium = 'own_metric'

def debug_states(pred,available_actions):

    for a,action in enumerate(available_actions):
        print('{} {}'.format(action,pred[a]))


def get_num_lines(file_path):

    fp = open(file_path,'r+')
    buf = mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def get_state_list(all_dialogues):
    '''
    Gets all states found in the data and returns the max len of the dialogues in the data

    :return:
    '''

    all_states = []
    max_dialogue_len = 0
    for dlg in all_dialogues:
        if len(all_dialogues[dlg]['dialogue']) > max_dialogue_len:
            max_dialogue_len = len(all_dialogues[dlg]['dialogue'])
        for state_def in [d['current_state'] for d in all_dialogues[dlg]['dialogue']]:
            all_states.append(state_def)

    state_counter = collections.Counter(all_states)
    if DEBUG:
        for k,v in sorted(state_counter.items()):
            print('{}: {}'.format(k,v))

    return list(set(all_states)),max_dialogue_len


def get_all_utterances(data_files):
    all_utterances = []

    for file in data_files.values():
        for turn in file['dialogue']:
            if 'user' in turn:
                all_utterances.append(turn['user'].lower())

    return all_utterances


def load_bow(data_files_dict=None,all_utterances=[]):

    if data_files_dict != None:
        if get_all_utterances == None:
            logging.error('Please defined the function collect utterances')
            sys.exit()
        all_utterances = []
        all_utterances += get_all_utterances(data_files_dict['train'])

    corpus_vect = CountVectorizer(binary=True)
    corpus_counts = corpus_vect.fit_transform(all_utterances)
    corpus_tfidf = TfidfTransformer(use_idf=False).fit(corpus_counts)
    return corpus_vect, corpus_counts.shape[1]

def data_to_h5(outputfile,input_data,target_data=[],action_mask=[]):

    if os.path.isfile(outputfile):
        logging.warning('{} already exists'.format(outputfile))
    else:
        data_f = h5py.File(outputfile,'w')
        if isinstance(input_data,dict):
            for f in input_data:
                data_f.create_dataset(f,data=input_data[f])
        else:
            data_f.create_dataset('features',data=input_data)
            if not isinstance(target_data,list):
                if not type(action_mask) == type([]):
                    data_f.create_dataset('action_mask', data=action_mask)
        if target_data.size:
            data_f.create_dataset('target',data=target_data)

        data_f.close()

def get_feature_size_from_data(file_name,action_mask=False,feat_sep=None,feature_key='features'):

    if not os.path.isfile(file_name):
        logging.error('File {} doesn\'t exist'.format(file_name))

    logging.info('Loading from {}'.format(file_name))
    data = h5py.File(file_name,'r')
    if action_mask:
        return data[feature_key].shape[0],data['features'].shape[1],data['action_mask'].shape[1]
    elif feat_sep:
        return data['target'].shape[0]
    else:
        return data[feature_key].shape

def load_data(data_files_list,
                  model_info,
                  cnn_dims_enb = None,
                  output_sep = False):

    # print(len(data_files_list), max_tensor_len, action_set_size)

    target_matrix = np.empty((len(data_files_list), model_info.max_dial_len, model_info. n_classes))
    matrices = {}
    for f, file in enumerate(data_files_list):
        data = h5py.File(file, 'r')
        if 'features' in data:
            logging.error('Old format not supported. Please separate data when extrating features.')
            sys.exit()

        target_matrix[f] = k_utils.to_categorical(data['target'], num_classes=model_info.n_classes)
        # action_mask_matrix[f] = data['action']
        if f == 0:
            for name in model_info.feature_set:
                if name == 'wrd_emb_gn' and cnn_dims_enb:
                    matrices[name] = np.empty((len(data_files_list), model_info.max_dial_len, 23, 300))
                else:
                    #try:
                    matrices[name] = np.empty((len(data_files_list), model_info.max_dial_len, data[name].shape[1]))
                    #except:
                    #    logging.info('{} dim {}'.format(name, data[name].shape[0]))
                    #    matrices[name] = np.empty((len(data_files_list), model_info.max_dial_len, data[name].shape[0]))

        for name in model_info.feature_set:
            matrices[name][f] = data[name]

    if output_sep:
        return matrices, target_matrix
    else:
        return np.concatenate(list(matrices.values()),axis=-1), target_matrix

def get_nclases(h5py_file_list,max_utt_len):

    classes = []

    for f in h5py_file_list:
        data = h5py.File(f)
        try:
            target_values = np.empty((max_utt_len,1))
            target_values[:,:] = data['target']
        except:
            logging.warning('{} has no targets'.format(f))
            continue
        for un in np.unique(target_values):
            if un not in classes:
                classes.append(un)

    return len(classes)

def train_model(output_dir,
                iteration,
                model_info,
                train_dialogues,
                validation_dialogues,
                dialogue_setting,
                action_set=None,
                log_dir=None):


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model_output = os.path.join(output_dir, '{}.hdf5'.format(iteration))

    if os.path.isfile(model_output):
        logging.warning(f'{model_output} already exists')
        return

    x_train, y_train = load_data(train_dialogues,
                                     model_info,
                                     output_sep=True)

    x_val, y_val = load_data(validation_dialogues,
                                 model_info,
                                 output_sep=True)

    if isinstance(x_train,dict):

        feat_sizes = {}
        for xd in x_train:
            if xd in model_info.feature_set:
                feat_sizes[xd] = x_train[xd].shape[2]

        model_info.feat_size = sum(list(feat_sizes.values()))
        model = keras_utils.configure_model(model_info, feat_sizes)

    else:

        if model_info.feat_size == 0:
            model_info.feat_size = x_train.shape[2]

        model = keras_utils.configure_model(model_info)

    keras_utils.debug_device()

    if model_info.stop_criterium == 'keras_checkpoint':

        checkpoint = callbacks.ModelCheckpoint(model_output,monitor='val_acc',verbose=0,save_best_only=True,save_weights_only=False,period=1)
        model.fit(x_train,y_train,epochs=model_info.epochs,batch_size=model_info.batch_size,validation_data=(x_val,y_val),callbacks=[checkpoint])

    else:

        turn_acc = 0.0
        best_epoch = -1

        for i in range(model_info.epochs):

            if isinstance(x_train, dict):

                model.fit(np.concatenate([x_train[f] for f in x_train if f in model_info.feature_set],axis=-1),
                          y_train, batch_size=model_info.batch_size,epochs=1,
                          validation_data=(np.concatenate([x_val[f] for f in x_val if f in model_info.feature_set],axis=-1), y_val),
                          verbose=1)

                prediction = model.predict(np.concatenate([x_val[f] for f in x_val if f in model_info.feature_set],axis=-1), batch_size=model_info.batch_size, verbose=1)
            else:
                model.fit(x_train,y_train,
                          batch_size=model_info.batch_size,
                          epochs=1,validation_data=(x_val, y_val),verbose=1)

                prediction = model.predict(x_val, batch_size=model_info.batch_size, verbose=1)

            acc, acc_replace, task_acc, dialogue_acc, \
            end_state_succ, correct_outcome, turns_m, situated_succ, perplex, d_perplex, \
            turns_m_all, acc_merged_updates, test_output, pred_output, collaborative_ts = compute_scores(y_val,
                                                                      x_val,
                                                                      prediction,
                                                                      dialogue_setting,
                                                                      action_set,
                                                                      validation_dialogues,
                                                                      model_info,
                                                                      'train',
                                                                      os.sep.join(log_dir.split(os.sep)[:-1]))

            print(f"Epoch {i + 1}/{model_info.epochs} - Accuracy {acc} - Accuracy (ent) {acc_replace} - Perplexity {perplex} - Task Success {task_acc}")
            if acc > turn_acc:
                logging.info("Improved turn accuracy, saving model from epoch {}".format(i))
                turn_acc = acc
                best_epoch = i
                print(f'saving weights in {model_output}')
                model.save_weights(model_output)

        logging.info('Best accuracy for {} achieved in epoch {}'.format(iteration,  best_epoch))
        keras_utils.clear_memory()

def test_model(model_info,
               model_path,
               test_dialogues,
               action_set,
               log_dir,
               dialogue_setting,
               lm_dacts=None):


    x_test, y_test = load_data(test_dialogues,
                                   model_info,
                                   output_sep=True)

    if isinstance(x_test,dict):

        feat_sizes = {}
        for xd in x_test:
            if xd in model_info.feature_set:
                feat_sizes[xd] = x_test[xd].shape[2]

        model_info.feat_size = sum(list(feat_sizes.values()))
        model = keras_utils.configure_model(model_info, feat_sizes)

    else:

        if model_info.feat_size == 0:
            model_info.feat_size = x_test.shape[2]

        model = keras_utils.configure_model(model_info)

    if os.path.isfile(model_path):
        logging.info(f"loading available model from {model_path}")
        model.load_weights(model_path)
    else:
        logging.error('No weigths file: {}'.format(model_path))
        input()

    if isinstance(x_test,dict):
        y_pred = model.predict(np.concatenate([x_test[f] for f in x_test if f in model_info.feature_set],axis=-1))
    else:
        y_pred = model.predict(x_test)

    if 'action_mask' in model_info.feature_set:
        acc_no_ent, acc_ent, mission_success, \
        dialogue_success, end_state_succ, corr_outcome, \
        turns_m, situated_succ, perplex, d_perplex, \
        acc_merged_updates, turns_m_all, \
        test_output, pred_output, collaborative_ts = compute_scores(y_test,
                                                x_test,
                                                y_pred,
                                                dialogue_setting,
                                                action_set,
                                                test_dialogues,
                                                model_info,
                                                log_dir = os.sep.join(log_dir.split(os.sep)[:-1]),
                                                action_mask_test=model_info.feature_set.index('action_mask'),
                                                lm_dacts=lm_dacts)

    else:
        acc_no_ent, acc_ent, mission_success, \
        dialogue_success, end_state_succ, corr_outcome,  \
        turns_m, situated_succ, perplex, d_perplex, \
        acc_merged_updates,turns_m_all, \
        test_output, pred_output, collaborative_ts = compute_scores(y_test,
                                                x_test,
                                                y_pred,
                                                dialogue_setting,
                                                action_set,
                                                test_dialogues,
                                                model_info,
                                                log_dir = os.sep.join(log_dir.split(os.sep)[:-1]),
                                                lm_dacts = lm_dacts)

    keras_utils.clear_memory()
    return acc_no_ent, acc_ent, mission_success,\
           end_state_succ, corr_outcome, situated_succ, turns_m, \
           perplex, d_perplex, turns_m_all, acc_merged_updates, \
           test_output, pred_output, collaborative_ts

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          only_existing_classes=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    if only_existing_classes:
        classes = [classes[c] for c in unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    create_plots(cm, classes, title)

    return cm, classes

def create_plots(cm,
                 y_classes,
                 title,
                 x_classes = None,
                 show_numbers=False,
                 cmap=plt.cm.Blues,
                 normalize = True,
                 vertical=False,
                 ):

    if x_classes is None:
        x_classes = y_classes

    is_square = len(y_classes) == len(x_classes)
    fig_size = (7.5, 15) if vertical else (6.5, 6.5) if is_square else (13,6.5)
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax_duv = make_axes_locatable(ax)
    cax = ax_duv.append_axes('top',size='2%' if vertical else '7%', pad='2%' if vertical else '5%')
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=x_classes, yticklabels=y_classes,
           ylabel='Predicted label' if vertical else 'True label',
           xlabel='True label' if vertical else 'Predicted label')
    cax.xaxis.set_ticks_position('top')

    #obj = plt.getp(ax, 'xticklabels')

    for t in ax.xaxis.get_ticklabels():
        if t._text in utils.action_types_dict['robot_updates'] or t._text == 'robot_updates':
            t.set_color('r')
        elif t._text in utils.action_types_dict['situation_updates'] or t._text == 'situation_updates':
            t.set_color('orange')
        elif t._text in utils.action_types_dict['actions'] or t._text == 'actions':
            t.set_color('g')
        elif t._text in utils.action_types_dict['interaction'] or t._text == 'interaction':
            t.set_color('b')
        elif t._text in utils.action_types_dict['request'] or t._text == 'request':
            t.set_color('k')
        else:
            t.set_color('c')

    for t in ax.yaxis.get_ticklabels():
        if t._text in utils.action_types_dict['robot_updates'] or t._text == 'robot_updates':
            t.set_color('r')
        elif t._text in utils.action_types_dict['situation_updates'] or t._text == 'situation_updates':
            t.set_color('orange')
        elif t._text in utils.action_types_dict['actions'] or t._text == 'actions':
            t.set_color('g')
        elif t._text in utils.action_types_dict['interaction'] or t._text == 'interaction':
            t.set_color('b')
        elif t._text in utils.action_types_dict['request'] or t._text == 'request':
            t.set_color('k')
        else:
            t.set_color('c')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    #plt.title(title, y=1.0)

    # Loop over data dimensions and create text annotations.
    if show_numbers:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
           for j in range(cm.shape[1]):
               ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()



def compute_scores(y_test,
                   x_test,
                   y_pred,
                   dialogue_setting,
                   action_set,
                   tst_files,
                   model_info,
                   mode = 'test',
                   log_dir=None,
                   action_mask_test=None,
                   lm_dacts=None):
    '''
    Computes scores for the orca data
    :param y_test:
    :param x_test:
    :param y_pred:
    :param rig_db:
    :param action_set:
    :param tst_files:
    :param log_dir:
    :param action_mask_test:
    :return:
    '''

    test = np.argmax(y_test,axis=-1)
    pred = np.argmax(y_pred,axis=-1)

    correct_prediction = []
    turn_acc_updates_merged = []
    mission_success = []
    end_state_success = []
    turns_mission_success = []
    turns_mission_success_succ_d = []
    dialogue_success = []
    correct_outcome = []
    situated_da_success = []
    d_perplex = []
    preplex_diff = []
    collaborative_task_success = []

    actions_test = []
    actions_pred = []

    if model_info.generate_output and mode != 'train':
        generate_output = '{}_{}'.format('.'.join(model_info.feature_set), model_info.config)
    else:
        generate_output = None

    # loop over dialogues
    for d in range(test.shape[0]):
        # we are to produce entity replacement results
        dialogue_id = tst_files[d].split(os.sep)[-1].split('.')[0]
        dialogue_log = json.load(open(utils.get_turn_file(dialogue_id,log_dir=log_dir,extension=f'shcn.data.json'), 'r'))
        gt_dialogue_success = utils.get_dialogue_success(dialogue_id,log_dir=log_dir)
        all_turns = dialogue_log['dialogue']

        if DEBUG:
            input(tst_files[d])

        if 'condition' in dialogue_log and dialogue_log['condition'] != 'mixed':
            condition = dialogue_log['condition']
        else:
            condition = 'user_plan'

        entity_replacement_success = []
        generated_subtask_sequence = set(['inspect'])

        if generate_output:
            generated_dialogue = {'condition': condition,
                                  'dialogue_id': dialogue_log['dialogue_id'],
                                  'model_info': model_info.__dict__,
                                  'turns': []}

            if not os.path.isdir(os.path.join(log_dir,'generated',generate_output)):
                os.makedirs(os.path.join(log_dir,'generated',generate_output))
            output_file_generated_dialogue = os.path.join(log_dir,'generated',generate_output,f"{dialogue_log['dialogue_id']}.json")

            if DEBUG:
                print(output_file_generated_dialogue)

        # getting the templates for each condition
        if condition == 'mixed':
            rig_db = dialogue_setting['user_plan'].rig_db
            interaction_manager = dialogue_setting['user_plan']
        else:
            rig_db = dialogue_setting[condition].rig_db
            interaction_manager = dialogue_setting[condition]

        #reinitialising dialogue
        interaction_manager.reset_dialogue()

        end_generated_dialogue = False # controlling if dialogue has reached final state
        turns_to_complete_mission_pred = test.shape[1] #max numbers of turns allowed
        turns_to_complete_mission_test = len(all_turns) #end of the actual dialogue

        d_actions_pred = []
        d_actions_test = []

        # loop over turns
        for t in range(test.shape[1]):

            if isinstance(x_test, list):
                mask_vector_test = x_test[1][d][t]
            elif isinstance(x_test, dict):
                for f in x_test:
                    if f.find('action_mask') > -1:
                        break
                mask_vector_test = x_test[f][d][t]
            else:
                mask_vector_test = x_test[d, t, :]

            if not np.all(mask_vector_test == 1):

                if DEBUG:
                    if 'prompt' in all_turns[t]:
                        print(f"S: {all_turns[t]['prompt']}")
                    if 'user' in all_turns[t]:
                        print(f"U: {all_turns[t]['user']}")

                if 'user' in all_turns[t] and generate_output is not None:
                    # gets the input for the user and updates the rig database
                    interaction_manager.turn['user'] = {}
                    interaction_manager.turn['user']['utt'] = all_turns[t]['user']
                    if 'nlu' in all_turns[t]['user']:
                        interaction_manager.get_nlu(all_turns[t]['user']['nlu'])

                try:
                    # sanity checks
                    if action_set[test[d, t]] != all_turns[t]['current_state']:
                        logging.error('Dialogue in the test vector is {}, but in the turns file is {}'.format(
                            action_set[test[d, t]], all_turns[t]['current_state']
                        ))
                except:
                    print(t)

                if all_turns[t]['current_state'] == 'inform_emergency_solved':
                    turns_to_complete_mission_test = t

                actions_test.append(test[d,t])
                d_actions_test.append(test[d,t])
                actions_pred.append(pred[d,t])
                d_actions_pred.append(pred[d,t])

                predicted_action = action_set[pred[d,t]]
                true_action = action_set[test[d,t]]

                rig_db.dynamic_mission_db['time_left'] = (rig_db.dynamic_mission_db['mission_time'] - all_turns[t]['time'])

                if DEBUG:
                    print(f"P: {predicted_action} [{true_action}]")

                if 'prompt' in all_turns[t]:
                    plain_prompt = utils.remove_punctuation(
                            all_turns[t]['prompt']).lower()  # cleaning original prompt for comparison
                else:
                    plain_prompt = None

                gen_turn_dict = {'d_act_pred': action_set[pred[d,t]],
                                 'd_act_gt': action_set[test[d,t]],
                                 'utt_gt': plain_prompt}

                if 'user' in all_turns[t]:
                    gen_turn_dict['user'] = all_turns[t]['user']

                # if predicted actions are different then the entity replacement will also fail
                if predicted_action != true_action:
                    entity_replacement_success.append(0)
                    situated_da_success.append(0)
                    gen_turn_dict['entity'] = False
                    gen_turn_dict['situated_da'] = False

                    if predicted_action in dialogue_setting[condition].nlg.status_updates and \
                        true_action in dialogue_setting[condition].nlg.status_updates:
                        turn_acc_updates_merged.append(1)
                    else:
                        turn_acc_updates_merged.append(0)

                    if action_set[pred[d,t]] in dialogue_setting['user_plan'].nlg.fixed_dialogue_acts:
                        if generate_output:
                            gen_turn_dict['utt_pred'] = action_set[pred[d,t]]
                    else:
                        candidate_utterances = nlg_utils.generate_utt_entities(action_set[pred[d,t]],interaction_manager)

                        if generate_output:
                            if len(candidate_utterances) == 0:
                                if DEBUG:
                                    print(action_set[pred[d, t]])
                                    input(condition)
                                gen_turn_dict['utt_pred'] = None
                            else:
                                gen_turn_dict['utt_pred'] = random.choice(candidate_utterances)

                # if action predicted is one of those which are not part of the template
                else:
                    turn_acc_updates_merged.append(1)
                    if predicted_action in ['user_turn','okay','activate.robot','deactivate.robot',
                                                 'holdon2seconds','gesture_emotion_neutral','actionperformed','repeat','BC',
                                                 'UNK','sorrycanyourepeatthat','yes','no']:

                        if generate_output:
                            gen_turn_dict['utt_pred'] = predicted_action
                            gen_turn_dict['entity'] = True

                        entity_replacement_success.append(1)

                    # if action is part of the template checks if the exact utterance can be reconstructed from the templates
                    else:

                        utt_in_template = False
                        candidate_utterances = nlg_utils.generate_utt_entities(predicted_action,interaction_manager)

                        for ca in candidate_utterances:
                            try:
                                ca = utils.remove_punctuation(ca).lower()
                                if ca == plain_prompt:
                                    utt_in_template = True
                                    break
                            except KeyError:
                                logging.warning(f'No prompt found in turn {t}')
                                break

                        if generate_output and len(candidate_utterances) > 0:
                            gen_turn_dict['utt_pred'] = ca

                        if utt_in_template:
                            entity_replacement_success.append(1)
                            gen_turn_dict['entity'] = True
                        else:
                            entity_replacement_success.append(0)
                            gen_turn_dict['entity'] = False
                            if DEBUG:
                                logging.debug('error in entity replacement for dialogue act {}'.format(action_set[pred[d,t]]))
                                utils.print_dict(candidate_utterances)
                                utils.print_dict(all_turns[t])
                                input()

                if t > 0:
                    if all_turns[t]['current_state'] in ['inform_moving','inform_robot_eta','inform_arrival',
                                                        'inform_inspection','inform_damage_inspection_robot',
                                                        'inform_returning_to_base','inform_robot_battery',                                                           'inform_robot_progress','inform_robot_velocity',
                                                        'inform_robot_status']:
                        if pred[d,t] == test[d, t]:
                            situated_da_success.append(1)
                            gen_turn_dict['situated_da'] = True
                        else:
                            situated_da_success.append(0)
                            gen_turn_dict['situated_da'] = False
                if generate_output:
                    generated_dialogue['turns'].append(gen_turn_dict)

            if action_set[pred[d,t]] == 'inform_emergency_status':
                generated_subtask_sequence.add('extinguish')
                interaction_manager.subtask = 'extinguish'

            if action_set[pred[d,t]] == 'inform_emergency_solved':
                generated_subtask_sequence.add('assess_damage')
                interaction_manager.subtask = 'assess_damage'
                turns_to_complete_mission_pred = t
                if generate_output:
                    generated_dialogue['turns_to_complete'] = t

            #print(generated_subtask_sequence)
            # if one of the two ending states is reached
            if action_set[pred[d,t]] in ['mission_timeout', 'inform_mission_completed'] and \
                    not end_generated_dialogue:
                #check if sequence of sub-dialogues fullfils the task needs
                end_generated_dialogue = True
                #end_state_success.append(1)
                if DEBUG:
                    utils.print_dict(list(generated_subtask_sequence))
                if set(generated_subtask_sequence) == set(['inspect', 'extinguish', 'assess_damage']):
                    mission_success.append(1)
                    if generate_output:
                        generated_dialogue['mission_success'] = True
                else:
                    mission_success.append(0)
                    if generate_output:
                        generated_dialogue['mission_success'] = False

                if action_set[pred[d,t]] == 'inform_mission_completed' and mission_success[-1] == 1 or\
                        action_set[pred[d,t]] == 'mission_timeout' and mission_success[-1] == 0:
                    end_state_success.append(1)
                    if generate_output:
                        generated_dialogue['end_state'] = True
                else:
                    end_state_success.append(0)
                    if generate_output:
                        generated_dialogue['end_state'] = False
                    #end_state_success.append(1)

        if not end_generated_dialogue:
            if set(generated_subtask_sequence) == set(['inspect', 'extinguish', 'assess_damage']):
                mission_success.append(1)
                if generate_output:
                    generated_dialogue['mission_success'] = True
            else:
                mission_success.append(0)
                if generate_output:
                    generated_dialogue['mission_success'] = False
            # if we reach the end of the dialogue without reaching one of the end states
            # dialogues are considered non-successful
            if generate_output:
                generated_dialogue['end_state'] = False
            end_state_success.append(0)

        if generate_output:
            generated_dialogue['relative_turns_on_task'] = turns_to_complete_mission_pred/turns_to_complete_mission_test

        if gt_dialogue_success is not None:
            if gt_dialogue_success == mission_success[-1]:
                correct_outcome.append(1)
                if generate_output:
                    generated_dialogue['correct_outcome'] = True
                    generated_dialogue['expected_outcome'] = gt_dialogue_success
                    generated_dialogue['achieved_outcome'] = mission_success[-1]
            else:
                correct_outcome.append(0)
                if generate_output:
                    generated_dialogue['correct_outcome'] = False
                    generated_dialogue['expected_outcome'] = gt_dialogue_success
                    generated_dialogue['achieved_outcome'] = mission_success[-1]

            if gt_dialogue_success:
                # if the gt dialogue is successful then check if the output of
                # the model is also success. Else give a 0
                if gt_dialogue_success == mission_success[-1]:
                    collaborative_task_success.append(1)
                else:
                    collaborative_task_success.append(0)
                # do not add anything to collaborative_task_success otherwise
        else:
            correct_outcome.append(0) #no meaning at all in this case

        turns_mission_success.append(turns_to_complete_mission_pred/turns_to_complete_mission_test)

        if mission_success[-1]:
            turns_mission_success_succ_d.append(turns_to_complete_mission_pred/turns_to_complete_mission_test)

        if sum(entity_replacement_success) == len(entity_replacement_success):
            if mission_success[-1] == 0:
                logging.error('Mission was successful')
            dialogue_success.append(1)
            if generate_output:
                generated_dialogue['dialogue_entity_success'] = True
        else:
            dialogue_success.append(0)
            if generate_output:
                generated_dialogue['dialogue_entity_success'] = False

        correct_prediction += entity_replacement_success
        #input()

        if generate_output:
            generated_dialogue['turn_accuracy'] = accuracy_score(d_actions_test,d_actions_pred)
            generated_dialogue['turn_accuracy_merged_updates'] = sum(turn_acc_updates_merged)/len(turn_acc_updates_merged)

        if lm_dacts:
            d_prob_pred = lm_dacts.p(' '.join([t['d_act_pred'] for t in generated_dialogue['turns']]))
            d_perplex.append(utils.perplexity(d_prob_pred))
            if generate_output:
                generated_dialogue['da_preplexity'] = utils.perplexity(d_prob_pred)
            d_prob_test = lm_dacts.p(' '.join([t['current_state'] for t in all_turns]))
            preplex_diff.append(utils.perplexity(d_prob_test) - utils.perplexity(d_prob_pred))
            if generate_output:
                generated_dialogue['da_preplexity_diff'] = utils.perplexity(utils.perplexity(d_prob_test) - utils.perplexity(d_prob_pred))
        else:
            d_perplex.append(0.0)
            preplex_diff.append(0.0)

        if generate_output:
            with open(output_file_generated_dialogue,'w') as dg_fp:
                json.dump(generated_dialogue, dg_fp, indent=2)

    if len(mission_success) != len(tst_files):
        utils.print_dict(all_turns)
        print(len(mission_success),len(end_state_success),len(correct_prediction))
        input()

    return accuracy_score(actions_test,actions_pred),\
           sum(correct_prediction)/len(correct_prediction),\
           sum(mission_success)/len(mission_success),\
           sum(dialogue_success)/len(dialogue_success),\
           sum(end_state_success)/len(end_state_success),\
           sum(correct_outcome)/len(correct_outcome),\
           sum(turns_mission_success)/len(turns_mission_success),\
           sum(situated_da_success)/len(situated_da_success),\
           sum(d_perplex)/len(d_perplex),\
           sum(preplex_diff)/len(preplex_diff),\
           sum(turn_acc_updates_merged)/len(turn_acc_updates_merged),\
           turns_mission_success_succ_d,\
           actions_pred,actions_test,\
           sum(collaborative_task_success) / len(collaborative_task_success) if len(collaborative_task_success) > 0 else np.nan