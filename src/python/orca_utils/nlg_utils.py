import collections
import logging
import math
import random
import re,os
from math import ceil

import itertools

import sys
import yaml

from orca_utils import rig_utils, state_utils, utils

DEBUG = False
MAX_FLOW_UTTERANCES = 13 #number of flow utterances that can be selected

class NLG():

    def __init__(self,states,condition, init_counter = False, expression_regex = r"\[(.*?)\]"):

        self.condition = condition
        self.states_nlg = {}
        self.states_with_no_transitions = []
        # dialogue acts which are not part of the template defined states
        self.fixed_dialogue_acts = ['repeat',
                                    'holdon2seconds',
                                    'sorrycanyourepeatthat',
                                    'ack_misunderstood',
                                    'okay',
                                    'idonthavethatinformationatthemoment',
                                    'info_not_available',
                                    'UNK',
                                    'no',
                                    'BC',
                                    'yes',
                                    'actionperformed',
                                    'gesture_emotion_neutral',
                                    'activate.robot',
                                    'deactivate.robot']

        # dialogue acts which will affect the state of the environment
        self.motion_acts = ['inform_moving']

        # gazebo updates
        self.robot_uptate_states = ['inform_robot_eta',
                                    'inform_robot_progress',
                                    'inform_robot_status',
                                    'inform_robot_velocity',
                                    'inform_robot_battery']

        self.time_winning_states = ['BC',
                                    'holdon2seconds']

        self.error_handling_states = ['request_repeat',
                                      'repeat',
                                      'idonthavethatinformationatthemoment']

        self.status_updates = self.robot_uptate_states + \
            ['inform_wind',
             'inform_time_left']

        self.init_states(states, init_counter)

        self.expression_regex = expression_regex

    def init_states(self,states, init_counter):

        if self.condition in states.split(os.sep):
            states_path = states
        else:
            states_path = os.path.join(states,self.condition)

        logging.info(f'Loading states from {states_path}')

        for root, dir, files in os.walk(states_path):
            for file in files:
                if file.endswith('yaml'):
                    logging.debug('Loading state information form file {}'.format(file))
                    state_name = file.rsplit('.',1)[0]
                    self.states_nlg[state_name] = state_utils.State(os.path.join(root, file), self.condition, init_counter)
                    if len(self.states_nlg[state_name].transitions) == 0:
                        self.states_with_no_transitions.append(state_name)

        if len(self.states_nlg.keys()) == 0:
            logging.critical('No states loaded')
        else:
            logging.debug(
                f'States loaded: {len(self.states_nlg.keys())}\n'
                f'Fixed states:  {len(self.fixed_dialogue_acts)})\n'
                f'Total states:  {len(self.states_nlg.keys()) + len(self.fixed_dialogue_acts)})')

        for fs in self.fixed_dialogue_acts:
            if fs in self.states_nlg:
                continue
            else:
                self.states_nlg[fs] = state_utils.PreDefState(fs,init_counter)
                self.states_with_no_transitions.append(fs)

    def print_states(self):

        for st in self.states_nlg:
            self.states_nlg[st].print_state_content()

    def print_transitions(self):

        for st in self.states_nlg:
            utils.print_dict(self.states_nlg[st].transitions_user_turn)
            utils.print_dict(self.states_nlg[st].transitions_system_turn)

    def write_transitions(self,output_dir):

        for st in self.states_nlg:
            state_dict = collections.OrderedDict({'name': self.states_nlg[st].name,
                          'common': {'formulations': self.states_nlg[st].utterances,
                                     'transition_states': {'system_turn': self.states_nlg[st].transitions_system_turn}}})

            if len(self.states_nlg[st].transitions_user_turn) > 0:
                state_dict['common']['transition_states']['user_turn'] = self.states_nlg[st].transitions_user_turn

            utils.print_dict(state_dict)

            utils.setup_yaml()
            with open(os.path.join(output_dir,'{}.yaml'.format(st)),'w') as of:
                yaml.dump(state_dict,of)


    def get_utterance(self, state_name, user_input_nlu = ''):

        return self.states_nlg[state_name].get_best_utterance(user_input_nlu)

def process_time(time_value):

    # fix so it trims seconds off
    if isinstance(time_value, str) and 'seconds' in time_value:
        time_value = time_value[:-7].strip()

    minutes, seconds = divmod(float(time_value), 60)
    if minutes > 1.0:
        if seconds == 1.0:
            return f'{minutes:2.0f} minutes and {seconds:2.0f} second'
        elif seconds > 1.0:
            return f'{minutes:2.0f} minutes and {seconds:2.0f} seconds'
        else:
            return f'{minutes:2.0f} minutes'
    elif minutes == 1.0:
        if seconds == 1.0:
            return f'{minutes:2.0f} minute and {seconds:2.0f} second'
        elif seconds > 1.0:
            return f'{minutes:2.0f} minute and {seconds:2.0f} seconds'
        else:
            return f'{minutes:2.0f} minute'
    else:
        seconds = ceil(seconds) #putting seconds to the upper boundary
        if seconds == 1.0 or seconds == 0.0:
            return '1 second'
        elif seconds > 1.0:
            return f'{seconds:2.0f} seconds'



def keys_available_set(comb_dict):

    keys_available = []
    for k in comb_dict.keys():
        key = re.sub('\{', '', k)
        key = re.sub('\}', '', key)
        if 'id' in key.split('_'):
            #skip id keys for this purpose
            continue
        keys_available.append(key)

    return set(keys_available)

def multiple_replace(sub_dict,utt):

    regex = re.compile('{}'.format('|'.join(map(re.escape,sub_dict.keys()))))

    return regex.sub(lambda mo: sub_dict[mo.string[mo.start():mo.end()]],utt)

def gen_combinations(values_dict,slots):

    combinations = []

    for comb in list(itertools.product(*values_dict.values())):
        plain_combo = ()
        plain_slots = []
        for idx,s in enumerate(comb):
            if isinstance(s,dict):
                for f in s:
                    plain_combo += (s[f],)
                    plain_slots.append(f)
            else:
                plain_combo += (s,)
                plain_slots.append(slots[idx])

        comb_dict = {}
        for i in range(len(plain_combo)):
            comb_dict['{{{}}}'.format(plain_slots[i])] = plain_combo[i]

        combinations.append(comb_dict)

    return combinations


def preProcess(utt, interaction_manager, expression_regex =r"\[(.*?)\]"):
    '''
    Removes gesture tags from the sentences before sending them to the wizard interface
    and replaces fields with values from the situation knowledge base
    :param utt:
    :return:
    '''

    gesture = []
    communicative_acts = []
    gesture_regex = re.compile(expression_regex)


    if DEBUG:
        print(f"filling slot values in {utt}")

    for gesture_match in re.finditer(gesture_regex,utt):
        for g in range(len(gesture_match.groups())):
            gesture_string = gesture_match.group(g)
            # remove pattern for the utterance
            if expression_regex.find('<') > -1:
                utt = re.sub(gesture_string, '', utt)
                utt = utt.strip()
            else:
                utt = re.sub(gesture_string[1:-1],'',utt)
                if expression_regex.find(r'\|') > -1:
                    utt = re.sub(r'\|','', utt)
                elif expression_regex.find(r"\[") > -1:
                    utt = re.sub(r'\[','',utt)
                    utt = re.sub(r'\]','',utt)
                else:
                    logging.error(f'expression regex not accepted {expression_regex}')
                    sys.exit()
                utt = utt.strip()
            gesture.append(gesture_string[1:-1])

    curly_braces_regex = re.compile(r"{(.*?)}")
    matches = re.finditer(curly_braces_regex,utt)
    slots_to_replace = set()
    if not curly_braces_regex.search(utt):
        if DEBUG:
            logging.debug('No field to replace')
        communicative_acts.append({'utterance': utt, 'gestures':';'.join(gesture)})
    else:
        for match in matches:
            for p in range(len(match.groups())):
                matched_field = match.group(p)
                slots_to_replace.add(matched_field[1:-1])

        if DEBUG:
            logging.debug('Fields to replace {}'.format(', '.join(slots_to_replace)))
        sub_dict = {f:[] for f in slots_to_replace}

        for slot in slots_to_replace:
            if 'wind' in slot.split('.'):
                variable = slot.split('.')[1]
                for condition in interaction_manager.situation_db['weather']:
                    if condition == 'wind':
                        sub_dict[slot].append(interaction_manager.situation_db['weather']['wind'][variable])
            elif 'spreading' in slot.split('.'):
                variable = slot.split('.')[1]
                if variable == 'object':
                    sub_dict[slot].append({slot: interaction_manager.situation_db['spreading'][variable]['name'],
                                           'location_id'.format(slot): interaction_manager.situation_db['spreading'][variable]['id']})
                else:
                    sub_dict[slot].append(interaction_manager.situation_db['spreading'][variable])
            elif slot in ['robot.eta','robot.battery_level','robot_type','robot.skill','robot.emergency_device',
                          'status','destination','robot.speed','robot.status','robot.progress','robot.location']:
                if 'robot.name' in slots_to_replace:
                    # this is treated together with the robot
                    continue

                if 'robot.skill' in slots_to_replace:
                    for rbt in interaction_manager.situation_db['robot']:
                        if rbt['active']:
                            if slot in rbt:
                                sub_dict[slot].append({'robot_id':rbt['id'],
                                                       'robot.skill':'{} and {}'.format(','.join(rbt[slot][:-1]),rbt[slot][-1])})
                else:
                    for rbt in interaction_manager.situation_db['robot']:
                        if rbt['active']:
                            try:
                                kb_slot_name = slot.split('.')[1]
                            except:
                                print(f'{slot} not available')
                                continue
                            if kb_slot_name in rbt:
                                if type(rbt[kb_slot_name]) == type([]):
                                    for i in rbt[kb_slot_name]:
                                        sub_dict[slot].append({slot:i, 'robot_id': rbt['id']})
                                else:
                                    if kb_slot_name == 'eta':
                                        sub_dict[slot].append({slot: process_time(rbt[kb_slot_name]), 'robot_id': rbt['id']})
                                    else:
                                        sub_dict[slot].append({slot: rbt[kb_slot_name], 'robot_id': rbt['id']})
            elif slot == 'robot.name':
                for item in interaction_manager.situation_db['robot']:
                    if item['active']:
                        if 'robot.emergency_device' in slots_to_replace and 'emergency_device' in item:
                            if 'robot.emergency_device' in sub_dict:
                                del sub_dict['robot.emergency_device']
                                for ed in item['emergency_device']:
                                    sub_dict[slot].append({'robot.name': item['name'], 'robot.emergency_device': ed, 'robot_id': item['id']})
                        else:
                            robot_dict = {'robot.name': item['name'], 'robot_id': item['id']}
                            if 'robot.eta' in slots_to_replace and 'eta' in item:
                                if 'robot.eta' in sub_dict:
                                    del sub_dict['robot.eta']
                                robot_dict['robot.eta'] = process_time(item['eta'])
                            if 'robot.battery_level' in slots_to_replace and 'battery_level' in item:
                                if 'robot.battery_level' in sub_dict:
                                    del sub_dict['robot.battery_level']
                                robot_dict['robot.battery_level'] = item['battery_level']
                            if 'robot_type' in slots_to_replace and 'robot_type' in item:
                                if 'robot_type' in sub_dict:
                                    del sub_dict['robot_type']
                                robot_dict['robot_type'] = item['robot_type']
                            if 'robot.progress' in slots_to_replace and 'progress' in item:
                                if 'robot.progress' in sub_dict:
                                    del sub_dict['robot.progress']
                                robot_dict['robot.progress'] = item['progress']
                            if 'robot.speed' in slots_to_replace and 'speed' in item:
                                if 'robot.speed' in sub_dict:
                                    del sub_dict['robot.speed']
                                robot_dict['robot.speed'] = item['speed']
                            if 'robot.location' in slots_to_replace:
                                if 'robot.location' in sub_dict:
                                    del sub_dict['robot.location']
                                if item['near_recovery_point']:
                                    robot_dict['robot.location'] = 'at the recovery point'
                                else:
                                    robot_dict['robot.location'] = 'away from the recovery point'
                            if 'robot.skill' in slots_to_replace and 'skill' in item:
                                if 'robot.skill' in sub_dict:
                                    del sub_dict['robot.skill']
                                robot_dict['robot.skill'] = '{} and {}'.format(', '.join(item['skill'][:-1]),item['skill'][-1])
                            if 'robot.status' in slots_to_replace:
                                if 'robot.status' in sub_dict:
                                    del sub_dict['robot.status']
                                #if 'destination' in sub_dict:
                                #    del sub_dict['destination']
                                if 'status' in item:
                                    # without gazebo, status might be given as a list
                                    if isinstance(item['status'],list):
                                        for info in item['status']:
                                            if isinstance(info, dict) and 'destination' in info:
                                                continue
                                                #robot_dict['destination'] = info['destination']
                                            else:
                                                robot_dict['robot.status'] = item['status'][0]
                                    else:
                                        robot_dict['robot.status'] = item['status']
                                else:
                                    robot_dict['robot.status'] = 'is ready'

                            if not set(set(robot_dict.keys())).issubset(set(slots_to_replace)) and robot_dict == {}:
                                continue
                            sub_dict[slot].append(robot_dict)

            elif slot == 'robots':
                # only robots included in the plan can be used if plan is followed
                robot_names = []
                for plan in interaction_manager.situation_db['plan']:
                    plans = list(plan.keys())
                    if len(plans) > 1:
                        random.shuffle(plans)
                plan_dict = interaction_manager.situation_db['plan'][plans[0] - 1]
                for pl in plan_dict:
                    for d in plan_dict[pl]:
                        if 'robots' in d:
                            for r in d['robots']:
                                #rbt = rig_utils.check_robot_index(r, interaction_manager.situation_db)
                                rbt = rig_utils.get_robot_by_id(r, interaction_manager.situation_db['robot'])
                                if rbt == None:
                                    continue
                                robot_names.append(rbt['name'])
                sub_dict[slot].append('{} and {}'.format(', '.join(robot_names[:-1]),robot_names[-1]))
            elif slot == 'object' or slot == 'area':
                if isinstance(interaction_manager.situation_db[slot], list):
                    for item in interaction_manager.situation_db[slot]:
                        sub_dict[slot].append({slot:item['name'], '{}_id': item['id']})
                else:
                    sub_dict[slot].append({slot:interaction_manager.situation_db[slot]['name'], 'location_id'.format(slot): interaction_manager.situation_db[slot]['id']})
            elif slot in ['mission_time', 'time_left']:
                sub_dict[slot].append(process_time(interaction_manager.situation_db[slot]))
            elif slot == 'robot_types':
                types_available = []
                for rbt in interaction_manager.situation_db['robot']:
                    if rbt['robot_type'] not in types_available:
                        types_available.append(rbt['robot_type'])
                if len(types_available) > 1:
                    sub_dict[slot].append('{} or {}'.format(', '.join(types_available[:-1]),types_available[-1]))
                else:
                    return []
            elif slot == 'robot_wait':
                for rbt in interaction_manager.situation_db['robot']:
                    if 'waiting' in rbt and rbt['waiting']:
                        sub_dict[slot].append(rbt['name'])
            elif slot == 'robot_moving':
                for rbt in interaction_manager.situation_db['robot']:
                    if rbt['status'] in ['is moving', 'is flying', 'is landing', 'is taking off']:
                        sub_dict[slot].append(rbt['name'])
            elif slot in interaction_manager.situation_db:
                if isinstance(interaction_manager.situation_db[slot], list):
                    for i in interaction_manager.situation_db[slot]:
                        if i not in sub_dict[slot]:
                            sub_dict[slot].append(i)
                else:
                    if interaction_manager.situation_db[slot] not in sub_dict[slot]:
                        sub_dict[slot].append(interaction_manager.situation_db[slot])
            else:
                logging.warning(f'{slot} not found in the situation db to replace in {utt}')
                print(interaction_manager.situation_db)
                return []

        for combo_dict in gen_combinations(sub_dict,list(slots_to_replace)):
            if keys_available_set(combo_dict) != slots_to_replace:
                #Slots with values are different than the slots to replace
                continue
            ca_dict = {'utterance': multiple_replace(combo_dict,utt), 'gestures': ';'.join(gesture)}
            if '{robot_id}' in combo_dict:
                ca_dict['robot_id'] = combo_dict['{robot_id}']
            if '{location_id}' in combo_dict:
                ca_dict['location_id'] = combo_dict['{location_id}']
            communicative_acts.append(ca_dict)

    return communicative_acts

def generate_utt_entities(action,interacion_manager):

    com_acts = []
    candidate_utterances = []

    if action not in interacion_manager.nlg_states:
        logging.warning(f'{action} not in template, for {interacion_manager.condition}')
    else:
        for utt in interacion_manager.nlg_states[action].utterances:
            com_acts += preProcess(utt, interacion_manager)

        for ca in com_acts:
            candidate_utterances.append(ca['utterance'])

    return candidate_utterances

