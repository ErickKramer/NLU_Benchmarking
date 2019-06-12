from rasa_nlu.model import Interpreter
import json

interpreter = Interpreter.load("gpsr/models/current/nlu")

message = "go to the kitchen"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
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

message = "move to the clutery drawer"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "navigate to the bedroom"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "put the noodles on the bookshelf"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "take the bottle from the table"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "grasp the cup on the sink"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "put the soap on the stove"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "bring me the bottle"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])

print('==================================')

message = "take the bottle from the table to the kitchen"
result = interpreter.parse(message)
print('------------------------------')
print('Message: ', result['text'])
print('------------------------------')
# Returns unicode values for all the fields
print('Intent: ', result['intent']['name'])
print('Confidence: ', result['intent']['confidence'])
print('------------------------------')
for entity_dict in result['entities']:
    print('Extractor: ', entity_dict['extractor'])
    print('Entity: ', entity_dict['entity'])
    print('Value: ', entity_dict['value'])
    print('Confidence: ', entity_dict['confidence'])
