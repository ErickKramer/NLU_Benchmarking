import os
import sys
import time
import yaml
from reader import read_sentences, read_expected_values
sys.path.append(os.path.abspath('../'))
from mbot_natural_language_processing.mbot_nlu.common.src.mbot_nlu.simple_phrase_divider import divide_sentence_in_phrases
from mbot_natural_language_processing.mbot_nlu.common.src.mbot_nlu.mbot_nlu_common import NaturalLanguageUnderstanding
from collections import Counter
import time

class MbotNluTest():
    def __init__(self,dataset):
        '''
        Sets up necessary elements for mbot
        '''
        if dataset == 'Ropod':
            yaml_dict = yaml.load(open('../mbot_natural_language_processing/mbot_nlu_training/ros/config/config_mbot_nlu_training_ropod.yaml'))['test_params']
        else:
            # Load the test parameters from the yaml file
            yaml_dict = yaml.load(open('../mbot_natural_language_processing/mbot_nlu_training/ros/config/config_mbot_nlu_training.yaml'))['test_params']

        debug = yaml_dict['debug']

        print(yaml_dict['available_intents'])

        classifier_path = yaml_dict['classifier_path']
        print("Using classifier ", classifier_path)
        wikipedia_vectors_path = yaml_dict['base_path']



        # NLU class and instance
        self.nlu = NaturalLanguageUnderstanding(classifier_path, wikipedia_vectors_path, dataset, debug=debug)

        # Initialize session
        self.nlu.initialize_session()

        print('\033[1;34m----------------------------------\033[0;37m')
        print('\033[1;32mnlu session is running\033[0;37m')
        print('\033[1;34m----------------------------------\033[0;37m')

    def test_mbot_nlu(self, sentences, expected_outputs,debug):
        '''
        Test of mbot model:
        - Send a sentence to the mbot model
        - Compare the output of mbot against the expected output value.
        '''

        # ---- Metrics parameters -----
        # Action detection metrics
        true_positive_intention = 0 # Number of intentions correctly classified
        true_negative_intention = 0 # Number of intentions correctly classified as Negative
        false_positive_intention = 0 # Number of intentions misclassified
        false_negative_intention = 0 # Number of intentions not classified

        # Full Command Recognition metrics
        true_positive_sentence = 0 # Number of sentences fully understood
        true_negative_sentence = 0 # Number of sentences correctly not recognized?
        false_positive_sentence = 0 # Number of sentences not fully understood
        false_negative_sentence = 0 # Number of sentences not recognized

        # Word Error Rate metrics
        actions_error_collection = []
        objects_error_collection = []
        people_error_collection = []
        locations_error_collection = []
        sentences_error_collection = []

        start_time = time.time()

        # Iterate over all the sentences in the inputs file
        for sentence_idx, sentence in enumerate(sentences):
            result = None

            # Flags lists needed for the metrics
            passed_intentions = []
            passed_slots = []

            if debug: print('\033[1;32m--------------------------\033[0;37m')

            if debug: print("Working with sentence -> ", sentence[0])

            # Divide the sentence into phrases, as mbot functions expects it
            phrases = divide_sentence_in_phrases(sentence[0])

            # Get the result from MBOT model
            try:
                result = self.nlu.process_sentence(phrases)
                if debug: print("Result from mbot is ", result)

            except:
                if debug: print("\033[1;31m Result not found for sentence {}\033[0;37m".format(sentence))
                # false_negative_intention += 1
                false_negative_sentence += 1
                continue # No further analysis of the sentence needed

            # Wait to receive results for the nlu model
            while type(result) != list:
                time.sleep(0.01)

            # Check if output list has at least one item
            if len(result) >= 1:
                pass
            else:
                if debug: print("No intention or slot found for the sentence {}\033[0;37m ".format(sentence))
                # false_negative_intention += 1
                false_negative_sentence += 1
                continue # No further analysis of the sentence needed

            if debug: print("The expected output is ", expected_outputs[sentence_idx])

            # Get the expected output and slot of sentence of each phrase
            for phrase_idx, expected_output in enumerate(expected_outputs[sentence_idx]):

                # Values extracted from the outputs file
                expected_intent = expected_output[0][0]
                expected_slots = expected_output[1]

                # Evaluates the intention
                try:
                    assert result[phrase_idx][0] == expected_intent
                    true_positive_intention += 1
                    passed_intentions.append(True)

                except:
                    if debug: print("\033[1;31m ----> Intention failed \033[0;37m")

                    # Check for empty intentions
                    try:
                        # Check if intention was not classified
                        _ = result[phrase_idx][0]
                        false_positive_intention += 1
                        passed_intentions.append(False)

                    except IndexError:
                        false_negative_intention += 1

                    actions_error_collection.append(expected_intent)


                # Evaluate the slots
                for slot_idx, slot in enumerate(expected_slots):

                    try:
                        assert result[phrase_idx][1][slot_idx] == slot
                        passed_slots.append(True)

                    except:
                        passed_slots.append(False)

                        # Switch case to add the error to the proper metric list
                        if slot[0] == 'object':
                            objects_error_collection.append(slot[1])

                        elif slot[0] == 'destination' or slot[0] == 'source':
                            locations_error_collection.append(slot[1])

                        elif slot[0] == 'person':
                            people_error_collection.append(slot[1])

                        elif slot[0] == 'sentence':
                            sentences_error_collection.append(slot[1])

                        else:
                            if debug: print('\033[1;31m ----> Slot not found \033[0;37m')
                            sentences_error_collection.append(slot[1])

            if all(passed_intentions) and all(passed_slots):
                true_positive_sentence += 1
                if debug: print("\033[1;36m Sentence passed! \033[0;37m")
                if debug: print('\033[1;32m--------------------------\033[0;37m')

            else:
                false_positive_sentence += 1
                if debug: print("\033[1;31m Sentence failed! \033[0;37m")
                if debug: print('\033[1;32m--------------------------\033[0;37m')

        # Calculate the precision of the action detection TP / (TP + FP)
        ac_precision = true_positive_intention / (true_positive_intention + false_positive_intention)

        # Calculate the recall of the action detection TP / (TP + FN)
        ac_recall = true_positive_intention / (true_positive_intention + false_negative_intention)

        # Calculate the F-measure of the action detection (Precision * Recall) / (Precision + Recall)
        ac_f_measure = (ac_precision * ac_recall) / (ac_precision + ac_recall)

        # Calculate the accuracy of the recognition of the full command
        fcr_accuracy = (true_positive_sentence + true_negative_sentence) / (true_positive_sentence + \
                                                                            true_negative_intention + \
                                                                            false_positive_sentence + \
                                                                            false_negative_sentence)

        # Calculating total testing time
        end = time.time() - start_time


        print('\033[1;32m==========================\033[0;37m')
        print('\033[1;32mTEST REPORT\033[0;37m')

        print('\033[1;32m==========================\033[0;37m')
        print('\033[1;34m AC & FCR METRICS \033[0;37m')
        print('\033[1;32m--------------------------\033[0;37m')
        print('\033[1;34m True positive intention is {} \033[0;37m'.format(true_positive_intention))
        print('\033[1;34m True negative intention is {} \033[0;37m'.format(true_negative_intention))

        print('\033[1;34m False positive intention is {} \033[0;37m'.format(false_positive_intention))
        print('\033[1;34m False negative intention is {} \033[0;37m'.format(false_negative_intention))

        print('\033[1;32m--------------------------\033[0;37m')

        print('\033[1;34m True positive sentence is {} \033[0;37m'.format(true_positive_sentence))
        print('\033[1;34m True negative sentence is {} \033[0;37m'.format(true_negative_sentence))

        print('\033[1;34m False positive sentece is {} \033[0;37m'.format(false_positive_sentence))
        print('\033[1;34m False negative sentece is {} \033[0;37m'.format(false_negative_sentence))

        print('\033[1;32m--------------------------\033[0;37m')
        print('\033[1;34m Action detection precision is {0:.2f} \033[0;37m'.format(ac_precision))

        print('\033[1;34m Action detection recall is {0:.2f} \033[0;37m'.format(ac_recall))

        print('\033[1;34m Action detection F-measure is {0:.2f} \033[0;37m'.format(ac_f_measure))

        print('\033[1;32m--------------------------\033[0;37m')
        print('\033[1;34m Full Command Recognition accuracy is {} \033[0;37m'.format(fcr_accuracy))

        print('\033[1;32m==========================\033[0;37m')
        print('\033[1;34m WER METRICS \033[0;37m')
        print('\033[1;32m--------------------------\033[0;37m')
        print('\033[1;34m Actions error rate: {} \033[0;37m'.format(Counter(actions_error_collection)))
        print('\033[1;34m Objects error rate: {} \033[0;37m'.format(Counter(objects_error_collection)))
        print('\033[1;34m Locations error rate: {} \033[0;37m'.format(Counter(locations_error_collection)))
        print('\033[1;34m People error rate: {} \033[0;37m'.format(Counter(people_error_collection)))
        print('\033[1;34m Sentences error rate: {} \033[0;37m'.format(Counter(sentences_error_collection)))

        print('\033[1;32m==========================\033[0;37m')

        print('\033[1;34m Mbot total evaluation time {0:.2f} seconds for {1} sentences and around {2:.2f} seconds per sentence  \033[0;37m'.format(end, len(sentences), end/len(sentences)))

        print('\033[1;32m==========================\033[0;37m')
        print('\033[1;32mTEST COMPLETE\033[0;37m')
        print('\033[1;32m==========================\033[0;37m')

    def tearDown(self):
        self.nlu.close_session()
        print('\033[1;31mnlu session is closed\033[0;37m')
if __name__ == '__main__':
    # env variables for tf and cuda
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    phrases = divide_sentence_in_phrases("tell me what is the largest object on the dresser")

    print(phrases)
    # Get the result from MBOT model
    mbotTest = MbotNluTest()
    result = mbotTest.nlu.process_sentence(phrases)
    print(result)
    mbotTest.tearDown()
