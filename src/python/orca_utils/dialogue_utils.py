import logging, time

class dialogue:

    def __init__(self, nlg, rig_db):

        self.rig_db = rig_db
        self.situation_db = rig_db.dynamic_mission_db
        self.nlg = nlg
        self.nlg_states = nlg.states_nlg

    def compute_time_left(self):
        self.situation_db['time_left'] = self.situation_db['mission_time'] - (time.time() - self.mission_start)

    def reset_dialogue(self):

        self.mission_start = time.time()

        self.state_history = []
        self.turn = {}
        self.subtask_travelled_states = []
        self.subtask = 'inspect'
        self.compute_time_left()