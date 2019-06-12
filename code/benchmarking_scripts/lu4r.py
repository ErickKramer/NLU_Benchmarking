import requests
import re
import string
from collections import Counter
import os
import sys
sys.path.append(os.path.abspath('../'))
from mbot_natural_language_processing.mbot_nlu.common.src.mbot_nlu.simple_phrase_divider import divide_sentence_in_phrases
import time

class LU4RTest():
    def __init__(self,dataset):
        self.url = 'http://127.0.0.1:9090/service/nlu'
        self.begin_curl_msg = 'hypo={\"hypotheses\":[{\"transcription\":\"'
        self.end_curl_msg = '\", \"confidence\":0.9,\"rank\":0}]}&entities={\"entities\":[]}'
        self.dataset = dataset

    def test_lu4r_nlu(self, sentences, expected_outputs, debug):
        '''
        Test of LU4R model:
        - Send a sentence to the LU4R model
        - Compare the output of LU4R against the expected output value.
        '''

        # ---- Metrics parameters ----
        # Action Detection metrics
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

            # Divide the sentence into phrases, to improve performance in multiple sentences
            phrases = divide_sentence_in_phrases(sentence[0])

            for phrase in phrases:

                # Construct the message to be send to LU4R
                full_command_msg = self.begin_curl_msg + phrase + self.end_curl_msg

                # Get the result from LU4R
                if 'you may find' not in phrase and 'you will find' not in phrase:
                    try:

                        tmp_res = requests.post('http://127.0.0.1:9090/service/nlu', data = full_command_msg)

                        # Remove end of line character
                        msg = tmp_res.text.strip()

                        if msg == 'NO FRAME(S) FOUND':
                            if debug: print('\033[1;31m No interpretation given for the command \033[0;37m')
                            false_negative_intention += 1
                            print('FN intention increased')
                            # false_negative_sentence += 1
                            result.append([])
                            continue # No further analysis of the phrase needed

                        if debug: print("The received phrase from LU4R is ", msg)

                        translated_msg = self.lu4r_mbot_translator(msg)

                        if debug: print('Translated phrase is ', translated_msg)

                        result.append(translated_msg[0])

                    except:
                        if debug: print('\033[1;31m Phrase not processed\033[0;37m')
                        false_negative_intention += 1
                        result.append([])
                        print('FN intention increased')

            if len(list(filter(None, result))) == 0:
                if debug: print('\033[1;31m Sentence not processed\033[0;37m')
                false_negative_sentence += 1
                continue # No further analysis of the sentence needed

            if debug: print("The expected output is ", expected_outputs[sentence_idx])
            if debug: print("The full output of LU4R is ", result)

            for phrase_idx, expected_output in enumerate(expected_outputs[sentence_idx]):

                # Values extracted from the outputs file
                expected_intent = expected_output[0][0]
                expected_slots = expected_output[1]

                # Evaluates the intention
                try:
                    assert result[phrase_idx][0][0] == expected_intent
                    passed_intentions.append(True)
                    true_positive_intention += 1
                    print('TP intention increased')

                except:
                    if debug: print("\033[1;31m ----> Intention failed \033[0;37m")

                    # Check for empty intentions
                    try:
                        # Check if intention was not classified
                        _ = result[phrase_idx][0][0]
                        passed_intentions.append(False)
                        false_positive_intention += 1
                        print('FP intention increased')

                    except IndexError:
                        # false_negative_intention += 1
                        pass

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

        # Calculate the precision of the action detection
        ac_precision = true_positive_intention / (true_positive_intention + false_positive_intention)

        # Calculate the recall of the action detection
        ac_recall = true_positive_intention / (true_positive_intention + false_negative_intention)

        # Calculate the F-measure of the action detection
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
        print('\033[1;34m LU4R total evaluation time {0:.2f} seconds for {1} sentences and around {2:.2f} per sentence  \033[0;37m'.format(end, len(sentences), end/len(sentences)))
        print('\033[1;32m==========================\033[0;37m')
        print('\033[1;32mTEST COMPLETE\033[0;37m')
        print('\033[1;32m==========================\033[0;37m')

    def lu4r_mbot_translator(self, msg):
        '''
        Translates the output of LU4R into the format used by the golden dataset
        '''
        inferred_sentence = []

        debug = True

        # Remove useless characters from the msg
        msg = re.sub('[\)"\n]','',msg)

        # Split the answer of the model into phrases (One per action)
        phrases = msg.split('#')
        # print('----> phrases ', phrases)

        for phrase in phrases:
            # Split the phrase into key parts
            try:
                # Get the frame from the phrase
                frame = re.split("[(]+", phrase)[0]

                # Remove the frame from the phrase
                phrase = phrase.lstrip(frame + '(')
                # print('---------> phrase ', phrase)


                # # Remove useless words
                # phrase = phrase.replace('a ', '')\
                #                .replace(' in ', '')\
                #                .replace(' on ', '')\
                #                .replace('to ','')\
                #                .replace('at ', '')\
                #                .replace('the ','')\
                #                .replace('for ', '')\
                #                .replace('from ','')\
                #                .strip()
                slots = phrase.split(',')
                # print('--------> SLOTS FOUND: ', slots)

                for idx,slot in enumerate(slots):
                    words = []
                    for w in slot.split(':')[1].split(' '):
                        if len(w) == 1:
                            w = w.replace('a', '')
                        elif len(w) == 2:
                            w = w.replace('to','').replace('at', '').replace('in', '').replace('on', '')
                        elif len(w) == 3:
                            w = w.replace('for', '').replace('the','')
                        elif len(w) == 4:
                            w = w.replace('from','')

                        if w != '':
                            words.append(w)
                        slot = slot.split(':')[0]+':'+' '.join(words)
                        slots[idx] = slot
                # print('--------> SLOTS FOUND: ', slots)
                # Collect the slots in the phrase


            except:
                if debug: print("Erroneous msg format")
                continue

            # Translates LU4R frames to MBOT Intention
            if frame == 'MOTION' or frame == 'ARRIVING':
                intention = 'go'

            elif frame == 'BRINGING' or \
                frame == 'GIVING' or \
                frame == 'MANIPULATION' or \
                frame == 'PLACING' or \
                frame == 'RELEASING' or \
                frame == 'TAKING':
                intention = 'take'

            elif frame == 'COTHEME':
                intention = 'follow'

            elif frame == 'LOCATING' or \
                frame == 'PERCEPTION_ACTIVE':

                intention = 'find'

            elif frame == 'ATTACHING':
                intention = 'attach'

            else:
                intention = 'other'

            slots_formatted = []
            if intention != 'other':
                # Translates slots into mbot format
                for slot in slots:
                    # Ignore empty slots
                    if slot == '': continue

                    if 'goal' in slot or \
                       'ground' in slot:
                        if intention == 'attach':
                            slot_type = 'object'
                        else:
                            slot_type = 'destination'

                    elif 'source' in slot:
                        slot_type = 'source'

                    elif ('theme' in slot and 'cotheme' not in slot) or \
                        'entity' in slot or \
                        'phenomenon' in slot:
                        slot_type = 'object'

                    elif 'recipient' in slot or \
                         'beneficiary' in slot or \
                         'cotheme' in slot:
                        slot_type = 'person'

                    elif 'source' in slot:
                        slot_type = 'source'

                    else:
                        slot_type = 'unknown'

                    slot_complement = slot.split(':')[1]

                    full_slot = (slot_type,slot_complement)

                    slots_formatted.append(full_slot)

            inferred_sentence.append([[intention], slots_formatted])

        return inferred_sentence
