from rasa_nlu.model import Interpreter
from collections import Counter
import json
import os
import sys
import time
sys.path.append(os.path.abspath('../'))
from mbot_natural_language_processing.mbot_nlu.common.src.mbot_nlu.simple_phrase_divider import divide_sentence_in_phrases

class RasaTest():
    def __init__(self, dataset):

        if dataset == 'Ropod':
            # Path to trained model
            self.interpreter = Interpreter.load("../rasa_models/ropod/models/current/nlu")
        else:
            # Path to trained model
            self.interpreter = Interpreter.load("../rasa_models/gpsr/models/current/nlu")

    def test_rasa_nlu(self, sentences, expected_outputs,debug):
        '''
        Test of rasa_nlu model:
        - Send a sentence to the interpreter
        - Compare the output of the model against the expected output value.
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
            result = []

            # Flags lists needed for the metrics
            passed_intentions = []
            passed_slots = []

            if debug: print('\033[1;32m--------------------------\033[0;37m')

            if debug: print("Working with sentence -> ", sentence[0])

            # Divide the sentence into phrases
            phrases = divide_sentence_in_phrases(sentence[0])

            for phrase in phrases:
                try:
                    tmp_res = self.interpreter.parse(phrase)

                    translated_msg = self.rasa_info_extractor(tmp_res)

                    result.append(translated_msg)

                except:
                    if debug: print("\033[1;31m Phrase not processed \033[0;37m")
                    false_negative_intention += 1
                    # false_negative_sentence += 1
                    continue # No further analysis of the phrase needed

            # Check if output list has at least one item
            if len(result) == 0:
                if debug: print('\033[1;31m Sentence not processed\033[0;37m')
                false_negative_sentence += 1
                continue

            if debug: print("The expected output is ", expected_outputs[sentence_idx])

            if debug: print('Output from rasa is ', result)
            # Get the expected output and slot of sentence of each phrase
            for phrase_idx, expected_output in enumerate(expected_outputs[sentence_idx]):

                # Values extracted from the outputs file
                expected_intent = expected_output[0][0]
                expected_slots = expected_output[1]

                # Evaluates the intention
                try:
                    assert result[phrase_idx][0][0] == expected_intent
                    passed_intentions.append(True)
                    true_positive_intention += 1

                except:
                    if debug: print("\033[1;31m ----> Intention failed \033[0;37m")

                    # Check for empty intentions
                    try:
                        # Check if intention was not classified
                        _ = result[phrase_idx][0]
                        passed_intentions.append(False)
                        false_positive_intention += 1

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
        print('\033[1;34m Action detection precision is {} \033[0;37m'.format(ac_precision))

        print('\033[1;34m Action detection recall is {} \033[0;37m'.format(ac_recall))

        print('\033[1;34m Action detection F-measure is {} \033[0;37m'.format(ac_f_measure))

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
        print('\033[1;34m RASA total evaluation time {0:.2f} seconds for {1} sentences and around {2:.2f} per sentence  \033[0;37m'.format(end, len(sentences), end/len(sentences)))
        print('\033[1;32m==========================\033[0;37m')
        print('\033[1;32mTEST COMPLETE\033[0;37m')
        print('\033[1;32m==========================\033[0;37m')

    def rasa_info_extractor(self, phrase):
        '''
        Extracts the needed information from the JSON answer of
        rasa_nlu
        '''
        intent = [phrase['intent']['name']]

        slots = []

        for entity_dict in phrase['entities']:
            slots.append((entity_dict['entity'], entity_dict['value']))

        output = [intent, slots]

        return output
        #[[['go'], [('destination', 'towel rail')]]]



if __name__ == '__main__':
    rasa_nlu = RasaTest('gpsr')
    message = "go to the kitchen and take the bottle from the table"
    result = rasa_nlu.interpreter.parse(message)
    _ = rasa_nlu.rasa_info_extractor(result)
    print('------------------------------')
    print('Message: ', result['text'])
    print('------------------------------')
    # # Returns unicode values for all the fields
    print('Intent: ', result['intent']['name'])
    print('Confidence: ', result['intent']['confidence'])
    print('------------------------------')
    for entity_dict in result['entities']:
        print('Extractor: ', entity_dict['extractor'])
        print('Entity: ', entity_dict['entity'])
        print('Value: ', entity_dict['value'])
        print('Confidence: ', entity_dict['confidence'])
    # print(json.dumps(result, indent=2))
    print('==================================')
    # result = rasa_nlu.interpreter.parse("dock to the table")
    # print('------------------------------')
    # print('Message: ', result['text'])
    # print('------------------------------')
    # # Returns unicode values for all the fields
    # print('Intent: ', result['intent']['name'])
    # print('Confidence: ', result['intent']['confidence'])
    # print('------------------------------')
    # for entity_dict in result['entities']:
    #     print('Extractor: ', entity_dict['extractor'])
    #     print('Entity: ', entity_dict['entity'])
    #     print('Value: ', entity_dict['value'])
    #     print('Confidence: ', entity_dict['confidence'])
