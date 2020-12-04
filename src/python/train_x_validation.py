import argparse,logging,os,sys
import arpa
from numpy import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

from orca_utils import hcn_utils, utils, rig_utils, nlg_utils, dialogue_utils

DEBUG = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains LSTM for an Hybrid Code Network using x-validation procedure')
    parser.add_argument('--datadir', '-d', type=str, help='Folder where the data is stored', required=True)
    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs used', default=12)
    parser.add_argument('--layer_sizes', '-l', type=int, nargs='+', help='Units per layer', default=[128])
    parser.add_argument('--simulation_environment', '-se', type=str,
                        help='directory physical environment of the simulated rig', default='requirements/objects/')
    parser.add_argument('--robot_db', '-rdb', type=str, help='directory with the robots available',
                        default='requirements/robots')
    parser.add_argument('--mission_file', '-m', type=str, help='mission file',
                        default='missions/fire_processing_unit_east_tower.yaml')
    parser.add_argument('--experimental-condition', dest='exp_cond', required=True)
    parser.add_argument('--states', '-s', type=str,
                        help='path to the directory where the nlg state templates are stored', default='states/')
    parser.add_argument('--batch_size', '-bs', type=int, help='number of batches used in the training procedure',
                        default=3)
    parser.add_argument('--dact_lm', '-dam', type=str, help='path to the lm trained with the dialogue acts')
    parser.add_argument('--stop_criterium', '-st', type=str, help='stop criterium', default='own_metric')
    parser.add_argument('--overfit', '-o', action='store_true', help='uses train as test set to check if overfits')
    parser.add_argument('--generate_dialogues', '-g', action='store_true', help='generate output dialogues')
    parser.add_argument('--features', '-f', type=str, nargs='+', help='features to be used',
                        default=['bow', 'previous_action', 'nlu'])
    parser.add_argument('--dataset_limit_train', '-dl', type=int, help='max number of instances in the train dataset',
                        default=None)
    parser.add_argument('--store_best', '-sb',action='store_true', help='if selected chooses own dialogue based on defined metrics',default=False)

    random.seed(42)

    args = parser.parse_args()

    rig_db = rig_utils.rig_db({'robots': args.robot_db, 'environment': args.simulation_environment, 'mission': args.mission_file})

    random.seed(42)

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.setLevel(logging.INFO)

    task_dir = os.path.join(args.datadir, args.exp_cond)
    features_dir = os.path.join(task_dir, '.'.join(args.features))

    model_info = hcn_utils.ModelInfo(args.batch_size, concat_depth=False)
    model_info.layers = args.layer_sizes
    model_info.epochs = args.epochs
    model_info.stop_criterium = args.stop_criterium
    model_info.generate_output = args.generate_dialogues

    model_info.config = 'VHCN_LSTM{}_epochs{}_cv'.format('_'.join(str(l) for l in args.layer_sizes), args.epochs)

    if args.features:
        args.features.sort()
        model_info.feature_set = args.features
        models_dir = os.path.join(args.datadir, model_info.config, '.'.join(args.features))
    else:
        for fragment in args.datadir.split(os.sep):
            if '.' in fragment:
                model_info.feature_set = fragment.split('.')
                args.features = model_info.feature_set
                break
        models_dir = os.path.join(args.datadir, model_info.config)

    print(model_info.feature_set)

    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    log_file_name = os.path.join(models_dir,'train.log')

    with open(log_file_name,'w') as fp:
        fp.write('test')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if args.states:
        dialogue_settings = {}
        for cond in ['user_plan','system_plan']:
            dialogue_settings[cond] = dialogue_utils.dialogue(nlg_utils.NLG(args.states,cond),
                                                                 rig_db)
    else:
        dialogue_settings = {}

    subsets_file_lists = {}

    for root, dir, files in os.walk(task_dir):
        if '.'.join(args.features) not in root.split(os.sep):
            continue
        folder = root.split(os.path.sep)[-1]
        if folder in ['train', 'dev']:
            key = 'train_cv' if folder == 'train' else 'val_cv'
            subsets_file_lists[key] = [os.path.join(root, feature_file) for feature_file in files]


    for subs in subsets_file_lists:
        if len(subsets_file_lists[subs]) == 0:
            logger.warning('No feature file found for {} subset'.format(subs))
            continue

        model_info.max_dial_len = hcn_utils.get_feature_size_from_data(subsets_file_lists[subs][0], feat_sep=True)

    if model_info.max_dial_len == 0:
        logger.error('No dialogue found')
        sys.exit()

    logger.info('Loading actions set')
    try:
        with open(os.path.join(task_dir,'orca_action_set.txt'),'rb') as lfp:
            available_actions = pickle.load(lfp)
    except:
        with open(os.path.join(task_dir,'orca_action_set.txt'), 'r') as lfp:
            available_actions = lfp.read().splitlines()

    if args.dact_lm:
        logger.debug('Loading DAct language model')
        dact_lm = arpa.loadf(args.dact_lm)[0]
        if DEBUG:
            input(dact_lm)
    else:
        dact_lm = None

    if DEBUG:
        utils.print_dict(available_actions)
        input()

    model_info.n_classes = len(available_actions)
    print(f'classes: {model_info.n_classes}')

    metrics = {'turn_accuracy': [],
               'turn_accuracy_ent': [],
               'turn_accuracy_usr': [],
               'turn_accuracy_mu': [],
               'situated_da_success': [],
               'da_perplexity': [],
               'diff_da_perplexity': [],
               'mission_success': [],
               'end_state_success': [],
               'correct_output': [],
               'avg_length_mission': [],
               'avg_length_mission_succ': []}

    outputs = {'y_true_all': [],
               'y_pred_all': []}

    # get speakers in the train_test_set
    all_speakers = [utils.get_participant_id(feat_file.split(os.sep)[-1].split('.')[0], args.datadir) for feat_file in subsets_file_lists['train_cv']] # original
    all_speakers = list(set(all_speakers))

    logger.info(f"training model for {len(all_speakers)} speakers, from {len(subsets_file_lists['train_cv'])} dialogue")

    for s,speaker in enumerate(all_speakers):

        test_files = [feat_file for feat_file in subsets_file_lists['train_cv'] if utils.get_participant_id(feat_file.split(os.sep)[-1].split('.')[0], args.datadir) == speaker]
        train_files = [feat_file for feat_file in subsets_file_lists['train_cv'] if utils.get_participant_id(feat_file.split(os.sep)[-1].split('.')[0], args.datadir) != speaker]

        logger.info('{}/{} folds training'.format(s+1,len(all_speakers)))

        model_output = os.path.join(models_dir, '{}.hdf5'.format(speaker))
        model_info.model_output = model_output
        model_info.training_samples = len(train_files)
        model_info.test_samples = len(test_files)
        if not os.path.isfile(model_output):
            logger.info(f'train model for speaker {speaker} from scratch')
            hcn_utils.train_model(models_dir,
                                  speaker,
                                  model_info,
                                  train_files,
                                  subsets_file_lists['val_cv'],
                                  dialogue_settings,
                                  action_set=available_actions,
                                  log_dir=task_dir)

        logger.info('Testing in {} samples'.format(len(test_files)))

        ta, ta_ent, m_s, \
        end_s, correct_output, situated_succ, turns_m, \
        perplex, d_perplex, \
        turns_m_all, ta_merged_updates, \
        y_test, y_pred   = hcn_utils.test_model(model_info,
                                                    model_output,
                                                    test_files,
                                                    available_actions,
                                                    task_dir,
                                                    dialogue_settings,
                                                    lm_dacts=dact_lm)


        metrics['turn_accuracy'].append(ta)
        metrics['turn_accuracy_ent'].append(ta_ent)
        metrics['turn_accuracy_mu'].append(ta_merged_updates)
        metrics['mission_success'].append(m_s)
        metrics['end_state_success'].append(end_s)
        metrics['correct_output'].append(correct_output)
        metrics['avg_length_mission'].append(turns_m)
        metrics['situated_da_success'].append(situated_succ)
        metrics['da_perplexity'].append(perplex)
        metrics['diff_da_perplexity'].append(d_perplex)
        metrics['avg_length_mission_succ'] += turns_m_all
        outputs['y_true_all'] += y_test
        outputs['y_pred_all'] += y_pred

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #hcn_utils.plot_confusion_matrix(np.array(outputs['y_true_all']), np.array(outputs['y_pred_all']), classes=available_actions,
    #                                title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    cm, classes = hcn_utils.plot_confusion_matrix(outputs['y_true_all'], outputs['y_pred_all'], classes=available_actions,
                                                  normalize=True, title='Normalised DAct confusion',
                                                  nly_existing_classes=True)

    with open(os.path.join(models_dir, 'actions.txt'),'w') as fp:
        fp.write('\n'.join(classes))
    np.save(os.path.join(models_dir, 'confusion_matrix.npy'),cm)

    plt.savefig(os.path.join(models_dir, f'normalised_confusion_matrix_{".".join(args.features)}.pdf'))
    plt.savefig(os.path.join(models_dir, f'normalised_confusion_matrix_{".".join(args.features)}.png'))

metrics['avg_length_mission_succ'] = list(filter(lambda a: a != 0.0, metrics['avg_length_mission_succ']))

utils.print_dict(model_info.__dict__)
input()

logger.info('Results of the {}-fold speaker out validation: '.format(len(all_speakers)))
logger.info('Turn accuracy: {:.4f}'.format(sum(metrics['turn_accuracy'])/len(metrics['turn_accuracy'])))
logger.info('Turn accuracy entity: {:.4f}'.format(sum(metrics['turn_accuracy_ent'])/len(metrics['turn_accuracy_ent'])))
logger.info('Turn accuracy sit {:.4f}'.format(sum(metrics['situated_da_success'])/len(metrics['situated_da_success'])))
logger.info('Turn accuracy mu {:.4f}'.format(sum(metrics['turn_accuracy_mu'])/len(metrics['turn_accuracy_mu'])))
logger.info('TS : {:.4f}'.format(sum(metrics['mission_success'])/len(metrics['mission_success'])))
logger.info('TS END: {:.4f}'.format(sum(metrics['end_state_success'])/len(metrics['end_state_success'])))
logger.info(f"Correct output: {sum(metrics['correct_output'])/len(metrics['correct_output']):.4f}")
logger.info('TRL {:.4f}'.format(sum(metrics['avg_length_mission'])/len(metrics['avg_length_mission'])))
logger.info('TRL Succ {:.4f}'.format(sum(metrics['avg_length_mission_succ'])/len(metrics['avg_length_mission_succ']) if len(metrics['avg_length_mission_succ']) > 0 else 0.0))
logger.info('DA perplexity {:.4f}'.format(sum(metrics['da_perplexity'])/len(metrics['da_perplexity'])))
logger.info('Diff DA perplexity {:.4f}'.format(sum(metrics['diff_da_perplexity'])/len(metrics['diff_da_perplexity'])))
