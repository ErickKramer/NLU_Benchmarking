import os
import sys
import time
import datetime
from reader import read_sentences, read_expected_values
from mbot import MbotNluTest
from rasa import RasaTest
from lu4r import LU4RTest
from ecg_text import RobotTextAgent

if __name__ == '__main__':
    # env variables for tf and cuda
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # Choose the dataset to test OPTIONS[Cat1, Cat2, Ropod]
    # dataset = 'Cat1'
    dataset = 'Cat2'
    # dataset = 'Ropod'

    # Debug flag
    debug = True

    # Load the corresponding dataset file
    if dataset == 'Cat1':
        filename_single_inputs = '../datasets/gpsr_cat1_single_inputs.txt'
        filename_multiple_inputs = '../datasets/gpsr_cat1_multiple_inputs.txt'
        filename_single_outputs = '../datasets/gpsr_cat1_single_outputs.txt'
        filename_multiple_outputs = '../datasets/gpsr_cat1_multiple_outputs.txt'
    elif dataset == 'Cat2':
        filename_single_inputs = '../datasets/gpsr_cat2_single_inputs.txt'
        filename_multiple_inputs = '../datasets/gpsr_cat2_multiple_inputs.txt'
        filename_single_outputs = '../datasets/gpsr_cat2_single_outputs.txt'
        filename_multiple_outputs = '../datasets/gpsr_cat2_multiple_outputs.txt'
    elif dataset == 'Ropod':
        filename_single_inputs = '../datasets/Ropod_single_inputs.txt'
        filename_multiple_inputs = '../datasets/Ropod_multiple_inputs.txt'
        filename_single_outputs = '../datasets/Ropod_single_outputs.txt'
        filename_multiple_outputs = '../datasets/Ropod_multiple_outputs.txt'

    sentences_single = read_sentences(filename_single_inputs)
    sentences_multiple = read_sentences(filename_multiple_inputs)
    expected_outputs_single = read_expected_values(filename_single_outputs,dataset)
    expected_outputs_multiple = read_expected_values(filename_multiple_outputs,dataset)

    print('\033[1;35m========================================================================================\033[0;37m')
    print('\033[1;35m DATASET: {} Starting at -> {} \033[0;37m'.format(dataset, datetime.datetime.now()))
    print('\033[1;35m========================================================================================\033[0;37m')
    print("Number of single sentences {}".format(len(sentences_single)))
    print("Number of multiple sentences {}".format(len(sentences_multiple)))
    print("Number of single expected_outputs {}".format(len(expected_outputs_single)))
    print("Number of multiple expected_outputs {}".format(len(expected_outputs_multiple)))

    # Model to evaluate [MBOT, LU4R, ECG, RASA]
    model = 'MBOT'
    # model = 'LU4R'
    # model = 'RASA'
    # model = 'ECG'

    if model == 'MBOT':

        mbotTest = MbotNluTest(dataset)
        print('\033[1;32m================================\033[0;37m')
        print('\033[1;32mTESTING MBOT SINGLE SENTENCE \033[0;37m')
        print('\033[1;32m================================\033[0;37m')
        mbotTest.test_mbot_nlu(sentences_single, expected_outputs_single,debug)
        print('\033[1;32m================================\033[0;37m')
        print('\033[1;32mTESTING MBOT MULTIPLE SENTENCES \033[0;37m')
        print('\033[1;32m================================\033[0;37m')
        mbotTest.test_mbot_nlu(sentences_multiple, expected_outputs_multiple,debug)

        mbotTest.tearDown()
    elif model == 'LU4R':
        '''
        Note:
        - In a separate window run LU4R jar file:
        * java -jar -Xmx1G lu4r-server-0.2.1.jar simple cfr en 9090
        '''
        lu4rTest = LU4RTest(dataset)
        print('\033[1;32m================================\033[0;37m')
        print('\033[1;32mTESTING LU4R SINGLE SENTENCE \033[0;37m')
        print('\033[1;32m================================\033[0;37m')
        lu4rTest.test_lu4r_nlu(sentences_single, expected_outputs_single, debug)
        print('\033[1;32m================================\033[0;37m')
        print('\033[1;32mTESTING LU4R MULTIPLE SENTENCES \033[0;37m')
        print('\033[1;32m================================\033[0;37m')
        lu4rTest.test_lu4r_nlu(sentences_multiple, expected_outputs_multiple, debug)

    elif model == 'RASA':
        rasaTest = RasaTest(dataset)
        # print('\033[1;32m================================\033[0;37m')
        # print('\033[1;32mTESTING RASA SINGLE SENTENCE \033[0;37m')
        # print('\033[1;32m================================\033[0;37m')
        # rasaTest.test_rasa_nlu(sentences_single, expected_outputs_single, debug)
        print('\033[1;32m================================\033[0;37m')
        print('\033[1;32mTESTING RASA MULTIPLE SENTENCES \033[0;37m')
        print('\033[1;32m================================\033[0;37m')
        rasaTest.test_rasa_nlu(sentences_multiple, expected_outputs_multiple, debug)

    elif model == 'ECG':
        '''
        Note:
        - Make sure to run in separate windows :
        * analyzer.sh
        * python3 src/main/robots/robots_ui.py ../ecg_grammars/compRobots.prefs AgentUI
        * python3 tester.py TextAgent
        '''
        ecgTest = RobotTextAgent(sys.argv[1:]) # TextAgent
        # ecgTest.prompt()
        # os.environ["ECG_FED"] = "FED1"
        # print('\033[1;32m================================\033[0;37m')
        # print('\033[1;32mTESTING ECG SINGLE SENTENCE \033[0;37m')
        # print('\033[1;32m================================\033[0;37m')
        # ecgTest.sentence_sender(sentences_single, expected_outputs_single, debug)
        print('\033[1;32m================================\033[0;37m')
        print('\033[1;32mTESTING ECG MULTIPLE SENTENCES \033[0;37m')
        print('\033[1;32m================================\033[0;37m')
        ecgTest.sentence_sender(sentences_multiple, expected_outputs_multiple, debug)
