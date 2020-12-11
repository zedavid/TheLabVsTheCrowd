import json,sys,logging,re
import string
import os
import pickle
from collections import OrderedDict

import numpy as np
import yaml

exclude = set(string.punctuation)
table = str.maketrans('','',string.punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))


def get_log_naming_pattern(nlu_config):

    return re.compile('(\w+).(\w+).(\w+).{}-all.events.json'.format(nlu_config))

def print_dict(dict):

    print(json.dumps(dict,indent=2))

def check_mentioned_robots(frame):

    robots = ['husky', 'huskey', 'anymal', 'animal', 'quadcopter', 'quad copter']
    active_robots = []

    for rbt in robots:
        if rbt in ' '.join(frame):
            active_robots.append(rbt)

    return active_robots

def remove_punctuation(text):

    text = re.sub("<.*?>", "", text)

    return text.translate(table).strip()

def get_log_file(dial_id,dialogue_log_files=None,log_dir=None,nlu=None):
    '''
    get path to log file based on log id or hdf5 file
    :param dial_id:
    :return:
    '''


    if dialogue_log_files:
        for f in dialogue_log_files:
            d_id = re.sub('\.','_',f.split(os.sep)[-1].split('-')[0])
            if nlu in d_id.split('_'):
                d_id = '_'.join(d_id.split('_')[:-1])
            if dial_id == d_id:
                #print(f)
                return f

    else:
        json_pattern = get_log_naming_pattern(nlu)
        for root,dir,files in os.walk(log_dir):
            for f in files:
                if 'test' in root.split(os.sep) or 'no_transcription_dialogues' in root.split(os.sep):
                    #logging.warning('Skipping {}'.format(root))
                    continue

                if re.match(json_pattern, f):
                    d_id = re.sub('\.', '_', f.split(os.sep)[-1].split('-')[0])
                    if nlu in d_id.split('_'):
                        d_id = '_'.join(d_id.split('_')[:-1])
                    if dial_id == d_id:
                        #print(f)
                        return os.path.join(root,f)

    logging.error('Couldn\'t find the log-file for {}'.format(dial_id))
    sys.exit()

def get_turn_file(dial_id,log_dir,extension='.turns'):

    for root,dir,files in os.walk(log_dir):
        for f in files:
            if f.endswith(extension):
                if f.find(dial_id) > -1:
                    return os.path.join(root,f)

    logging.error('Couldn\'t find the turns-file for {}'.format(dial_id))
    sys.exit()

def get_dialogue_success(dial_id, log_dir):

    dialogue = None

    for root,dir,files in os.walk(log_dir):
        if 'generated' in root.split(os.sep):
            continue
        for f in files:
            if f.endswith('shcn.data.json'):
                continue
            if f.endswith('json') and dial_id == f.split('.')[0]:
                with open(os.path.join(root,f),'r') as dfp:
                    dialogue = json.load(dfp)
                break

    if dialogue is None:
        return None

    for evt in dialogue['logs']:
        if evt['event'] == 'post_task_analysis':
            return evt['data']['mission_successful']

def intersect_lists(list1, list2):
    return list(set(list1) & set(list2))

def diff_lists(list1, list2):
    return list(set(list1) - set(list2))

def get_participant_id(dialogue_id, data_dir):

    json_file = os.path.join(data_dir,f"{dialogue_id}.shcn.data.json")

    with open(json_file,'r') as fp:
        dialogue_json = json.load(fp)

    return dialogue_json['user_id']



def get_condition(dialogue_id):

    dialogue_id_pattern = re.compile('(\w+)_(\d+)_(\w+)')

    if re.match(dialogue_id_pattern,dialogue_id):
        mission, id, mode = re.match(dialogue_id_pattern,dialogue_id).groups()
        return mode
    else:
        input(dialogue_id)



def clean_pre_defined(utterance):
    '''
    Given an utterance from the pre-defined set remove space and other no standard characters
    :param utterance:
    :return:
    '''

    utt = re.sub(', ', '_', utterance)
    utt = re.sub(' ','_',utt)
    utt = re.sub('\'','',utt)
    utt = regex.sub('',utt)

    return utt.lower()

def process_dialogue_state():

    return

def get_dialogue_states(file_id,log_dir,action_set):

    # rebuilding folder identifier
    dialogue_id = file_id.split(os.sep)[-1].split('.')[0]
    if 'mixed' in dialogue_id.split('_'):
        dialogue_id, condition = dialogue_id.rsplit('_',1)
        mission_id, base_name, id = dialogue_id.rsplit('_',2)
        folder_id = '{}.{}_{}.{}'.format(mission_id,base_name,id,condition)
    else:
        mission_id, condition, plan = dialogue_id.rsplit('_',2)
        scenario_id, user, id = mission_id.rsplit('_',2)
        folder_id = '{}.{}_{}.{}_{}'.format(scenario_id,user,id,condition,plan)

    log_dir_name = os.path.join(log_dir,
                                folder_id)

    if not os.path.isdir(log_dir_name):
        log_dir_name = os.path.join(log_dir, 'pilots',
                                    folder_id)

    if os.path.isdir(log_dir_name):
        for file in os.listdir(log_dir_name):
            if file.endswith('.slurk.dlg'):
                with open(os.path.join(log_dir_name,file), 'r') as dlg_fp:
                    dialogues = json.load(dlg_fp)
                    return [action_set.index(d['current_state']) if d['current_state'] in action_set else 'UNK' for d in dialogues['dialogue']],folder_id
    else:
        logging.error('Original dialogue state couldn\'t be found for dialogue {}'.format(log_dir_name))
        sys.exit()

    return [],folder_id


def perplexity(prob):

    return np.power(2,-prob*np.log(prob))


def setup_yaml():
  """ https://stackoverflow.com/a/8661021 """
  represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
  yaml.add_representer(OrderedDict, represent_dict_order)


def jsonify(b_str,replace='"'):

    try:
        in_word_double_quote = re.compile('[a-zA-Z0-9]"[a-zA-Z0-9]')

        json_str = re.sub('b\'',replace,b_str)
        json_str = re.sub('b"',replace,json_str)
        json_str = re.sub('\'',replace,json_str)
        for pattern in re.findall(in_word_double_quote,json_str):
            outpatt = re.sub('"','\'',pattern)
            json_str = re.sub(pattern,outpatt,json_str)
        json_str = re.sub('"\{','{',json_str)
        json_str = re.sub('}"',"}",json_str)
    except:
        #case where the string those
        return b_str

    #print(json_str)

    return json_str

action_types_dict = {
    'robot_updates': [
        'inform_robot_crash', 'inform_robot_wait', 'inform_robot_eta',
        'inform_robot_progress', 'inform_robot_status', 'inform_robot_available',
        'inform_robot_battery','inform_robot_not_available','inform_robot_velocity',
        'inform_robot_capabilities', 'inform_robot_location'],
    'situation_updates': [
        'inform_time_left','inform_alert_emergency',
        'inform_emergency_status', 'inform_risk_spreading',
        'inform_emergency_solved','inform_risk', 'inform_mission_completed',
        'inform_plan_selected', 'mission_timeout', 'inform_wind',
        'inform_no_risk_spreading', 'inform_symptom', 'inform_action',
        'inform_error', 'inform_battery_loss', 'inform_abort_mission'],
    'actions': [
        'inform_moving','inform_arrival','inform_inspection','inform_returning_to_base',
        'inform_activate_emergency_shutdown','inform_emergency_action',
        'inform_damage_inspection','inform_deactivate_emergency_shutdown',
        'activate.robot', 'deactivate.robot'],
    'interaction': [
        'okay','holdon2seconds','BC','intro_hello','gesture_emotion_neutral',
        'sorrycanyourepeatthat','yes','no','idonthavethatinformationatthemoment',
        'request_attention','actionperformed', 'info_not_available', 'db_query',
        'inform_emergency_robot', 'inform_damage_inspection_robot',
        'request_repeat', 'start', 'confirm_availability', 'ack_misunderstood'],
    'request': [
        'request_pa_announcement','request_robot_inspect_damage',
        'request_robot_emergency','request_robot_type', 'request_grounding_location'],
    'unknown': ['UNK']
}