from orca_utils import utils, rig_utils
import requests, logging, re

nlu_server = 'http://127.0.0.1:7785/predict'
#nlu_server = 'http://137.195.243.7:7785/predict'

DEBUG = False

class NLU_feat():

    def __init__(self,robot_db,rig_db):

        self.robots = robot_db
        self.objects = rig_db
        self.nlu_vect = self.get_default_vector()


    def get_default_vector(self):

        nlu_vect = {}

        # frame semantics
        nlu_vect['sending'] = 0
        nlu_vect['motion'] = 0
        nlu_vect['putting_out_fire'] = 0
        nlu_vect['inspecting'] = 0
        nlu_vect['perception_active'] = 0
        nlu_vect['using'] = 0
        nlu_vect['being_located'] = 0
        nlu_vect['emptying'] = 0
        nlu_vect['bringing'] = 0

        # dialogue acts
        nlu_vect['instruction'] = 0
        nlu_vect['request_info'] = 0
        nlu_vect['disfluency'] = 0
        nlu_vect['inform'] = 0
        nlu_vect['acknowledgment'] = 0
        nlu_vect['request_action'] = 0
        nlu_vect['opening'] = 0
        nlu_vect['thanking'] = 0
        nlu_vect['reject'] = 0
        nlu_vect['apology'] = 0
        nlu_vect['closing'] = 0

        #robots
        robot_types = set()
        for rb in self.robots:
            nlu_vect['theme_{}'.format(rb['id'])] = 0
            nlu_vect['cotheme_{}'.format(rb['id'])] = 0
            robot_types.add(rb['robot_type'])

        #robot type
        for rb_t in robot_types:
            nlu_vect['theme_{}'.format(rb_t)] = 0
            nlu_vect['cotheme_{}'.format(rb_t)] = 0

        #objects
        for obj in self.objects:
            nlu_vect[obj['id']] = 0

        return nlu_vect

    def get_nlu_vector(self,nlu):

        self.nlu_vect = self.get_default_vector()

        for d_act in nlu['dialogue_acts']:
            if d_act['dialogue_act'].lower() in self.nlu_vect:
                self.nlu_vect[d_act['dialogue_act'].lower()] = 1
            else:
                input(f"{d_act['dialogue_act']} not defined")

            for frame in nlu['frame_semantics']:
                if frame['frame'].lower() in self.nlu_vect:
                    self.nlu_vect[frame['frame'].lower()] = 1
                else:
                    # frame will not be considered since it is not among the relevant ones
                    logging.debug(f"{frame['frame']} ignored")
                    continue

                if frame['frame'] in ['Sending','Motion']:
                    if 'frame_elements' in frame:
                        theme_tokens = []
                        goal_tokens = []
                        cotheme_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Theme':
                                theme_tokens += elements['tokens']
                                # entity_tracker.clear_robots()
                            if elements['frame_element'] == 'Goal':
                                goal_tokens += elements['tokens']
                            if elements['frame_element'] == 'Cotheme':
                                cotheme_tokens += elements['tokens']

                        if theme_tokens != []:
                            theme_tokens = utils.intersect_lists(theme_tokens,d_act['tokens'])
                            self.get_robot_features(theme_tokens, nlu)

                        if goal_tokens != []:
                            goal_tokens = utils.intersect_lists(goal_tokens,d_act['tokens'])
                            self.get_location_features(goal_tokens, nlu)

                        if cotheme_tokens != []:
                            cotheme_tokens = utils.intersect_lists(cotheme_tokens,d_act['tokens'])
                            self.get_robot_features(cotheme_tokens, nlu, 'cotheme')

                if frame['frame'] == 'Putting_out_fire':
                    if 'frame_elements' in frame:
                        place_tokens = []
                        agent_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Place':
                                place_tokens += elements['tokens']
                            if elements['frame_element'] == 'Agent':
                                agent_tokens += elements['tokens']

                        if place_tokens != []:
                            place_tokens = utils.intersect_lists(place_tokens,d_act['tokens'])
                            self.get_location_features(place_tokens, nlu)

                        if agent_tokens != []:
                            agent_tokens = utils.intersect_lists(agent_tokens,d_act['tokens'])
                            self.get_robot_features(agent_tokens, nlu)

                if frame['frame'] == 'Inspecting':
                    if 'frame_elements' in frame:
                        ground_tokens = []
                        inspector_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Ground':
                                ground_tokens += elements['tokens']
                            if elements['frame_element'] == 'Inspector':
                                inspector_tokens += elements['tokens']

                        if ground_tokens != []:
                            ground_tokens = utils.intersect_lists(ground_tokens, d_act['tokens'])
                            self.get_location_features(ground_tokens, nlu)

                        if inspector_tokens != []:
                            inspector_tokens = utils.intersect_lists(inspector_tokens, d_act['tokens'])
                            self.get_robot_features(inspector_tokens, nlu)

                if frame['frame'] == 'Perception_active':
                    lexical_unit_lemmas = get_lemma_tokens(frame['lexical_unit'],nlu['tokens'])
                    if 'hear' in lexical_unit_lemmas and 'see' not in lexical_unit_lemmas:
                        #hear is always related to somehting that the user did not understand
                        continue
                    if 'frame_elements' in frame:
                        agent_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Perceiver_agentive':
                                agent_tokens += elements['tokens']

                        if agent_tokens != []:
                            agent_tokens = utils.intersect_lists(agent_tokens,d_act['tokens'])
                            self.get_robot_features(agent_tokens, nlu)

                if frame['frame'] == 'Using':
                    if 'frame_elements' in frame:
                        for elements in frame['frame_elements']:

                            if elements['frame_element'] == 'Instrument':
                                self.get_robot_features(utils.intersect_lists(elements['tokens'],d_act['tokens']), nlu)

                if frame['frame'] == 'Being_located':
                    if 'frame_elements' in frame:
                        theme_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Theme':
                                theme_tokens += elements['tokens']

                        if theme_tokens != []:
                            theme_tokens = utils.intersect_lists(theme_tokens,d_act['tokens'])
                            self.get_robot_features(theme_tokens, nlu)

                if frame['frame'] == 'Emptying':
                    if 'frame_elements' in frame:
                        place_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Place':
                                place_tokens += elements['tokens']

                        if place_tokens != []:
                            place_tokens = utils.intersect_lists(place_tokens,d_act['tokens'])
                            self.get_location_features(place_tokens, nlu)

                if frame['frame'] == 'Bringing':
                    if 'frame_elements' in frame:
                        theme_tokens = []
                        for elements in frame['frame_elements']:
                            if elements['frame_element'] == 'Theme':
                                theme_tokens += elements['tokens']

                            if theme_tokens != []:
                                theme_tokens = utils.intersect_lists(theme_tokens,d_act['tokens'])
                                self.get_robot_features(theme_tokens, nlu)


    def get_robot_features(self, tokens, nlu, func='theme'):

        device_name = get_surface_tokens(tokens, nlu['tokens'])

        if len(device_name.split()) > 1:
            if 'ner' in nlu and nlu['ner'] != []:
                robot_tokens = []
                for ne in nlu['ner']:
                    if ne['ne'] == 'PERSON':
                        robot_tokens += utils.intersect_lists(ne['tokens'], tokens)
                    if ne['ne'] == 'B-CARDINAL:':
                        robot_tokens += utils.intersect_lists(ne['tokens'], tokens)
                device_name = get_surface_tokens(robot_tokens, nlu['tokens'])
                robot_db_entry = rig_utils.get_robot_by_name(device_name, self.robots)
                if robot_db_entry != None:
                    self.nlu_vect['{}_{}'.format(func,robot_db_entry['id'])] = 1
                else:
                    robot_db_entries = rig_utils.get_robot_by_type(device_name, self.robots)
                    if robot_db_entries != []:
                        for rbt in robot_db_entries:
                            self.nlu_vect['{}_{}'.format(func,rbt['type'])] = 1


    def get_location_features(self,location_tokens,nlu):

        location_name = get_surface_tokens(location_tokens, nlu['tokens'])

        if location_name == '':
            return

        simulation_db_entries = rig_utils.get_location_by_name(location_name, self.objects)
        if simulation_db_entries != []:
            for l in simulation_db_entries:
                self.nlu_vect[l['id']] = 1


def get_surface_tokens(element_tokens, nlu_tokens, remove_pron_lemmas = False):

    surfaces = []

    for token in nlu_tokens:
        if token['id'] in element_tokens:
            if token['pos'] not in ['DET','ADP']:
                if remove_pron_lemmas and token['lemma'] != '-PRON-' and token['pos'] != 'PART' or not remove_pron_lemmas:
                    surfaces.append(token['surface'])

    return ' '.join(surfaces)

def get_lemma_tokens(element_tokens, nlu_tokens):

    lemmas = []

    for token in nlu_tokens:
        if token['id'] in element_tokens:
            lemmas.append(token['lemma'])

    return lemmas


def extract_user_dialogue_acts(nlu):

    if 'dialogue_acts' in nlu:
        return [da['dialogue_act'] for da in nlu['dialogue_acts']]
    else:
        return []


def get_robot_name(nlu, tokens, dialogue_manager):

    robots = []

    device_name = get_surface_tokens(tokens, nlu['tokens'])

    if len(device_name.split()) > 1:
        if 'ner' in nlu and nlu['ner'] != []:
            robot_tokens = []
            for ne in nlu['ner']:
                if ne['ne'] == 'PERSON':
                    robot_tokens += utils.intersect_lists(ne['tokens'], tokens)
                if ne['ne'] == 'B-CARDINAL:':
                    robot_tokens += utils.intersect_lists(ne['tokens'], tokens)
            device_name = get_surface_tokens(robot_tokens, nlu['tokens'])

    robot_db_entry = rig_utils.get_robot_by_name(device_name, dialogue_manager.rig_db.dynamic_mission_db['robot'])

    if robot_db_entry != None:
        robot_db_entry['active'] = True
        robots.append(robot_db_entry)
    else:
        robot_db_entries = rig_utils.get_robot_by_type(device_name, dialogue_manager.rig_db.dynamic_mission_db['robpt'])
        if robot_db_entries != []:
            for rbt in robot_db_entries:
                rbt['active'] = True
                robots.append(rbt)

    return robots

def get_location_name(nlu, goal_tokens, dialogue_manager):

    selected_location = ''

    location_name = get_surface_tokens(goal_tokens, nlu['tokens'])
    sim_db = rig_utils.get_location_by_name(location_name, dialogue_manager.rig_db.dynamic_mission_db)

    if sim_db != []:
        gazebo_ids = []
        for l in sim_db:
            gazebo_ids.append(l['gazebo_id'])

        if len(list(set(gazebo_ids))) != 1:
            logging.error('Possible locations {}'.format(','.join(gazebo_ids)))
        else:
            selected_location = gazebo_ids[0]

    return selected_location

def extract_semantic_frames(nlu,dialogue_manager):

    nlu_dict = {'d_act': [], 'sem': []}

    for d_act in nlu['dialogue_acts']:
        nlu_dict['d_act'].append(d_act['dialogue_act'].lower())

    for frame in nlu['frame_semantics']:
        nlu_dict['sem'].append(frame['frame'].lower())
        if frame['frame'] in ['Sending','Motion','Being_located']:
            if 'frame_elements' in frame:
                theme_tokens = []
                goal_tokens = []
                cotheme_tokens = []
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Theme':
                        theme_tokens += elements['tokens']
                    if elements['frame_element'] == 'Goal':
                        goal_tokens += elements['tokens']
                    if elements['frame_element'] == 'Cotheme':
                        cotheme_tokens += elements['tokens']

                if theme_tokens != []:
                    theme_tokens = utils.intersect_lists(theme_tokens, d_act['tokens'])
                    nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(theme_tokens,nlu['tokens']),
                                                                            dialogue_manager.rig_db.dynamic_mission_db)

                if goal_tokens != []:
                    goal_tokens = utils.intersect_lists(goal_tokens, d_act['tokens'])
                    nlu_dict['location'] = get_location_name(nlu,goal_tokens,dialogue_manager)

                if cotheme_tokens != []:
                    cotheme_tokens = utils.intersect_lists(cotheme_tokens, d_act['tokens'])
                    nlu_dict['robot'] += rig_utils.parse_sentence_for_robots(get_surface_tokens(cotheme_tokens,nlu['tokens']),
                                                                                                dialogue_manager.rig_db.dynamic_mission_db)

        if frame['frame'] == 'Putting_out_fire':
            if 'frame_elements' in frame:
                place_tokens = []
                agent_tokens = []
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Place':
                        place_tokens += elements['tokens']
                    if elements['frame_element'] == 'Agent':
                        agent_tokens += elements['tokens']

                if place_tokens != []:
                    place_tokens = utils.intersect_lists(place_tokens, d_act['tokens'])
                    nlu_dict['location'] = get_location_name(nlu, place_tokens, dialogue_manager)

                if agent_tokens != []:
                    agent_tokens = utils.intersect_lists(agent_tokens, d_act['tokens'])
                    nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(agent_tokens,nlu['tokens']),
                                                                            dialogue_manager.rig_db.dynamic_mission_db)

        if frame['frame'] == 'Inspecting':
            if 'frame_elements' in frame:
                ground_tokens = []
                inspector_tokens = []
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Ground':
                        ground_tokens += elements['tokens']
                    if elements['frame_element'] == 'Inspector':
                        inspector_tokens += elements['tokens']

                if ground_tokens != []:
                    ground_tokens = utils.intersect_lists(ground_tokens, d_act['tokens'])
                    nlu_dict['location'] = get_location_name(nlu, ground_tokens, dialogue_manager)

                if inspector_tokens != []:
                    inspector_tokens = utils.intersect_lists(inspector_tokens, d_act['tokens'])
                    nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(inspector_tokens,nlu['tokens']),
                                                                                               dialogue_manager.rig_db.dynamic_mission_db)

        if frame['frame'] == 'Perception_active':
            lexical_unit_lemmas = get_lemma_tokens(frame['lexical_unit'],nlu['tokens'])
            if 'hear' in lexical_unit_lemmas and 'see' not in lexical_unit_lemmas:
                #hear is always related to somehting that the user did not understand
                continue
            if 'frame_elements' in frame:
                agent_tokens = []
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Perceiver_agentive':
                        agent_tokens += elements['tokens']

                if agent_tokens != []:
                    agent_tokens = utils.intersect_lists(agent_tokens,d_act['tokens'])
                    nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(agent_tokens,nlu['tokens']),
                                                                            dialogue_manager.rig_db.dynamic_mission_db)

        if frame['frame'] == 'Using':
            if 'frame_elements' in frame:
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Instrument':
                        instrument_tokens = utils.intersect_lists(elements['tokens'],d_act['tokens'])
                        nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(instrument_tokens,nlu['tokens']),
                                                                                dialogue_manager.rig_db.dynamic_mission_db)

        if frame['frame'] == 'Emptying':
            if 'frame_elements' in frame:
                place_tokens = []
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Place':
                        place_tokens += elements['tokens']

                if place_tokens != []:
                    place_tokens = utils.intersect_lists(place_tokens, d_act['tokens'])
                    nlu_dict['location'] = get_location_name(nlu, place_tokens, dialogue_manager)

        if frame['frame'] == 'Bringing':
            if 'frame_elements' in frame:
                theme_tokens = []
                for elements in frame['frame_elements']:
                    if elements['frame_element'] == 'Theme':
                        theme_tokens += elements['tokens']

                    if theme_tokens != []:
                        theme_tokens = utils.intersect_lists(theme_tokens, d_act['tokens'])
                        nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(theme_tokens,nlu['tokens']),
                                                                                dialogue_manager.rig_db.dynamic_mission_db)

        if frame['frame'] == 'Being_in_category':
            if 'frame_elements' in frame:
                item_tokens = []
                for elements in frame['frame_elements']:
                    item_tokens += elements['tokens']

                if item_tokens != []:
                    item_tokens = utils.intersect_lists(item_tokens, d_act['tokens'])
                    nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(item_tokens, nlu['tokens']),
                                                                            dialogue_manager.rig_db.dynamic_mission_db)

                    surface_tokens = get_surface_tokens(item_tokens, nlu['tokens'],remove_pron_lemmas=True)

                    if nlu_dict['robot'] != []:
                        # when the robot is mentioned
                        robot_names = rig_utils.parse_sentence_for_robot_name(surface_tokens,
                                                                             dialogue_manager.rig_db.dynamic_mission_db)
                        item = surface_tokens
                        for r in robot_names:
                            item = re.sub(r,'',item)
                        nlu_dict['item'] = item.strip()
                    else:
                        nlu_dict['item'] = surface_tokens

        if frame['frame'] == 'Being_located':
            if 'frame_elements' in frame:
                theme_tokens = []
                for elements in frame['frame_elements']:
                    theme_tokens += elements['tokens']

                if theme_tokens != []:
                    theme_tokens = utils.intersect_lists(theme_tokens, d_act['tokens'])
                    robots = rig_utils.parse_sentence_for_robots(get_surface_tokens(theme_tokens,nlu['tokens']),
                                                                 dialogue_manager.rig_db.dynamic_mission_db)
                    if len(robots) > 0:
                        nlu_dict['robot'] = robots
                    else:
                        nlu_dict['location'] = get_location_name(nlu, theme_tokens, dialogue_manager)

        if frame['frame'] == 'Telling':
            # "[can you] tell me the <robot_info>?" parser
            if 'frame_elements' in frame:
                message_tokens = []
                for elements in frame['frame_elements']:
                    message_tokens += elements['tokens']

                if message_tokens != []:
                    message_tokens = utils.intersect_lists(message_tokens, d_act['tokens'])
                    nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(get_surface_tokens(message_tokens, nlu['tokens']),
                                                                            dialogue_manager.situation_db)
                    surface_tokens = get_surface_tokens(message_tokens, nlu['tokens'], remove_pron_lemmas=True)
                    if nlu_dict['robot'] != []:
                        # when robot is mentioned
                        print(surface_tokens)
                        robot_names = rig_utils.parse_sentence_for_robot_name(surface_tokens, dialogue_manager.situation_db)
                        item = surface_tokens
                        for r in robot_names:
                            item = re.sub(r,'',item)
                        nlu_dict['item'] = item.strip()
                    else:
                        nlu_dict['item'] = surface_tokens

    for d_act in nlu['dialogue_acts']:
        if d_act['dialogue_act'] in ['INFORM','INSTRUCTION'] and nlu['frame_semantics'] == []:
            content = get_surface_tokens(d_act['tokens'], nlu['tokens'])
            nlu_dict['robot'] = rig_utils.parse_sentence_for_robots(content,dialogue_manager.rig_db.dynamic_mission_db)
            nlu_dict['item'] = parse_for_items(content)
        else:
            logging.debug(f"dialogue act {nlu_dict['d_act']} not in above listed or empty frame semantic output")
            # avoid printing the whole thing
            if DEBUG:
                utils.print_dict(nlu['dialogue_acts'])
                utils.print_dict(nlu['frame_semantics'])
                print(nlu['sentence'])

    if DEBUG:
        utils.print_dict(nlu_dict)

    return nlu_dict

def parse_for_items(surface_realisation):

    if 'eta' in surface_realisation.split():
        return 'eta'

    if 'battery' in surface_realisation.split():
       return 'battery'

    if 'progress' in surface_realisation.split():
       return 'progress'

    if any([item in ['velocity','speed'] for item in surface_realisation.split()]):
        return 'speed'

    if 'status' in surface_realisation.split():
        return 'status'

    return None

def extract_minimal_nlu(dialogue_manager):

    turn = dialogue_manager.turn

    if 'trans' not in turn:
        user_utterance = turn['user']['utt']
    else:
        user_utterance = turn['trans']

    if utils.remove_punctuation(user_utterance) in ['yes', 'yeah', 'okay']:
        return {'sem':['acknowledgment']}
    elif utils.remove_punctuation(user_utterance) in ['no']:
        return {'sem':['reject']}
    elif utils.remove_punctuation(user_utterance) in ['uav']:
        #forcing nlu for the uav case
        return {'d_act':['instruction'],
                'robot': rig_utils.parse_sentence_for_robots(user_utterance,dialogue_manager.rig_db.dynamic_mission_db)}
    elif turn['user']['nlu'] == 'no_nlu':
        if DEBUG:
            utils.print_dict(turn)
        logging.error('No NLU input')
        input()
    else:
        nlu_dacts = extract_user_dialogue_acts(turn['user']['nlu'])
        if nlu_dacts == ['ACKNOWLEDGMENT']:
            return {'sem':['acknowledgment']}
        else:
            return extract_semantic_frames(turn['user']['nlu'],dialogue_manager)

    return None


def get_nlu_from_model(text):
    '''
    ### NOTE ###
    Please remember to launch the server before running this
    The server can be launched using the the following command

    python server.py -o json
    (Running Andrea's NLU server)
    ############

    :param text:
    :return:
    '''

    r = requests.post(nlu_server,json={"sentence":text})
    if r.status_code == 200:
        return r.json()
    else:
        logging.warning('Unable to parse the sentence')
    return {}
