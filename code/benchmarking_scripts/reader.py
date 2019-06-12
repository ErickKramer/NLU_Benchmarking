import re
def read_sentences(file_inputs):
    '''
    Extract the input sentences from the inputs.txt file
    '''
    # Input sentences
    sentences = []
    with open(file_inputs) as fp:
        for line in fp:
            # Remove empty lines and lines containing comments, i.e. lines starting with #
            if '#' not in line and line != '\n':
                sentences.append([line.strip('\n')])

    return sentences

def read_expected_values(file_outputs,dataset):
    # Expected outputs
    if dataset == 'Ropod':
        available_slots = ['person', 'object', 'source', 'destination', 'position']
        available_intents = ['go', 'attach', 'detach', 'push', 'find', 'guide', 'follow']
    else:
        available_slots = ['person', 'object', 'source', 'destination', 'sentence']
        available_intents = ['answer', 'find', 'follow', 'guide', 'take', 'tell', 'go', 'meet']

    expected_outputs = []

    with open(file_outputs) as fp:
        for line in fp:
            # skip commented and empty lines
            if '#' in line or line == '\n': continue

            # print("----> Sentence: ", line)

            # strip end chars and split
            line = line.rstrip().split()
            # print("----> Cleaned sentence: ", line)

            # Get the index of the intentions in the line
            intentions_idx = [idx for intent in available_intents for idx,word in enumerate(line) if word == intent]

            # Sort the indeces in decreasing order
            intentions_idx.sort(reverse=True)

            # Get the phrases in the line
            phrases = []

            for idx in intentions_idx:
                phrases.append(line[idx:])
                del line[idx:]
            phrases = phrases[::-1]

            expected_outputs_phrase = []

            for phrase in phrases:
                # print("----> phrase: ", phrase)
                #intent extraction
                intent = phrase.pop(0)

                # extracting the slot index from the current phrase
                slot_idx_list = []

                # for slot in available_slots:
                #     try: slot_idx_list.append(phrase.index(slot))
                #     except: continue
                slot_idx_list = [idx for slot in available_slots for idx,word in enumerate(phrase)
                                if word == slot and phrase[idx-1] != 'the' and phrase[idx-1] != slot]
                # print('Slot indices ', slot_idx_list)

                # Sorting the slot indexes
                slot_idx_list = sorted(slot_idx_list, key=int)

                # last element index
                slot_idx_list.append(len(phrase))

                #slots extraction and appending to the current intent
                slots = []

                for v, w in zip(slot_idx_list, slot_idx_list[1:]):
                    # slots are extracted according to their indexes from previous search
                    slot = (phrase[v].lower(), ' '.join(phrase[v+1:w]))

                    # appending to the last item in the list (which is the current intent)
                    slots.append(slot)

                # append intent and slots
                expected_outputs_phrase.append([[intent], slots])
            # print('---> expected output ', expected_outputs_phrase)
            expected_outputs.append(expected_outputs_phrase)
    return expected_outputs

if __name__ == '__main__':
    # Choose the dataset to test OPTIONS[Cat1, Cat2, Ropod]
    dataset = 'Cat1'

    # Load the corresponding dataset file
    if dataset == 'Cat1':
        filename_single_inputs = '../datasets/gpsr_cat1_single_inputs.txt'
        filename_multiple_inputs = '../datasets/gpsr_cat1_multiple_inputs.txt'
        filename_single_outputs = '../datasets/gpsr_cat1_single_outputs.txt'
        filename_multiple_outputs = '../datasets/gpsr_cat1_multiple_outputs.txt'
    elif dataset == 'Cat2':
        filename_inputs = '../datasets/gpsr_cat2_inputs.txt'
        filename_outputs = '../datasets/gpsr_cat2_outputs.txt'
    elif dataset == 'Ropod':
        filename_inputs = '../datasets/Ropod_inputs.txt'
        filename_outputs = '../datasets/Ropod_outputs.txt'

    sentences_single = read_sentences(filename_single_inputs)
    # sentences_multiple = read_sentences(filename_multiple_inputs)
    expected_outputs_single = read_expected_values(filename_single_outputs)
    # expected_outputs_multiple = read_expected_values(filename_multiple_outputs)

    print("Number of single sentences {}".format(len(sentences_single)))
    # print("Number of multiple sentences {}".format(len(sentences_multiple)))
    print("Number of single expected_outputs {}".format(len(expected_outputs_single)))
    # print("Number of multiple expected_outputs {}".format(len(expected_outputs_multiple)))
