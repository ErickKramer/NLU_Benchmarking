#!/usr/bin/env python3

import random
import yaml
import msgpack
import numpy as np
from sklearn.utils import resample

# load parameters from yaml
# ================================================================================================================
yaml_dict = yaml.load(open('../../../../../ros/config/config_mbot_nlu_training.yaml'))['intent_train']
random_state = eval(yaml_dict['resample_random_state'])

# params for balancing individual structures
# ================================================================================================================
# number of types of different structured sentences in each of intent classes
n_struct = {'go': 44, 'take': 35, 'find': 37, 'answer': 4, 'tell': 12, 'guide': 36, 'follow': 28, 'meet': 28}

# number of samples per structe required enough to make balances data
n_samples_per_intent = int(yaml_dict['n_examples']/len(n_struct))

# data slider. bigger the value, bigger the number of sentences with complex structures(eg: grasp to mia at the kitchen the bottle from the bed room)
# but bigger the repeatation of sentences with smaller structure(eg: go to the kitchen)
data_slider = yaml_dict['data_slider']

# data for creating sentences eg: [names, objects]
# ================================================================================================================
# objects
# ================================================================================================================
objects_a = ['kleenex', 'whiteboard cleaner', 'cup', 'snack', 'cereals bar', 'cookie', 'book', 'pen', 'notebook', 'laptop', 'tablet', 'charger',
                         'pencil', 'peanut', 'biscuit', 'candy', 'chocolate bar', 'chewing gum', 'chocolate egg', 'chocolate tablet', 'donuts', 'cake', 'pie',
                         'peach', 'strawberry', 'blueberry', 'blackberry', 'burger', 'lemon', 'lemon', 'banana', 'watermelon', 'pepper', 'pear', 'pizza',
                         'yogurt', 'drink', 'beer', 'coke', 'sprite', 'sake', 'toothpaste', 'cream', 'lotion', 'dryer', 'comb', 'towel', 'shampoo', 'soap',
                         'cloth', 'sponge', 'toothbrush', 'container', 'glass', 'can', 'bottle', 'fork', 'knife', 'bowl',
                         'tray', 'plate', 'newspaper', 'magazine']

objects_an = ['almond', 'onion', 'orange', 'apple']

objects_the = ['cookies', 'almonds', 'book', 'pen', 'notebook', 'laptop', 'tablet', 'charger', 'pencil', 'chips', 'senbei', 'pringles',
                           'peanuts', 'biscuits', 'crackers', 'candies', 'chocolate bar', 'manju', 'mints', 'chewing gums', 'chocolate egg', 'chocolate tablet',
                           'donuts', 'cake', 'pie', 'food', 'peach', 'strawberries', 'grapes', 'blueberries', 'blackberries', 'salt', 'sugar', 'bread', 'cheese',
                           'ham', 'burger', 'lemon', 'onion', 'lemons', 'apples', 'onions', 'orange', 'oranges', 'peaches', 'banana', 'bananas', 'noodles',
                           'apple', 'paprika', 'watermelon', 'sushi', 'pepper', 'pear', 'pizza', 'yogurt', 'drink', 'milk', 'juice', 'coffee', 'hot chocolate',
                           'whisky', 'rum', 'vodka', 'cider', 'lemonade', 'tea', 'water', 'beer', 'coke', 'sprite', 'wine', 'sake', 'toiletries', 'toothpaste',
                           'cream', 'lotion', 'dryer', 'comb', 'towel', 'shampoo', 'soap', 'cloth', 'sponge', 'toilet paper', 'toothbrush', 'container', 'containers',
                           'glass', 'can', 'bottle', 'fork', 'knife', 'bowl', 'tray', 'plate', 'newspaper', 'magazine', 'rice','kleenex', 'whiteboard cleaner', 'cup']

objects_some = ['snacks', 'cookies', 'almonds', 'books', 'pens', 'chips', 'pringles', 'magazines', 'newspapers', 'peanuts', 'biscuits',
                            'crackers', 'candies', 'mints', 'chewing gums', 'donuts', 'cake', 'pie', 'food', 'strawberries', 'grapes', 'blueberries',
                            'blackberries', 'salt', 'sugar', 'bread', 'cheese', 'ham', 'lemons', 'apples', 'onions', 'oranges', 'peaches', 'bananas',
                            'noodles', 'paprika', 'watermelon', 'sushi', 'pepper', 'pizza', 'yogurt', 'drink', 'milk', 'juice', 'coffee', 'hot chocolate',
                            'whisky', 'rum', 'vodka', 'cider', 'lemonade', 'tea', 'water', 'beer', 'coke', 'sprite', 'wine', 'sake', 'toilet paper',
                            'containers', 'glasses', 'cans', 'bottles', 'forks', 'knives', 'bowls', 'trays', 'plates', 'lemon', 'rice', 'cups']

objects_a_piece_of = ['apple', 'lemon', 'cake', 'pie', 'bread', 'cheese', 'ham', 'watermelon', 'sushi', 'pizza']

objects_a_cup_of = ['juice', 'rice', 'milk', 'coffee', 'hot chocolate', 'cider', 'lemonade', 'tea', 'water', 'beer']

objects_a_can_of = ['juice', 'kleenex', 'red bull', 'cider', 'iced tea', 'beer', 'coke', 'sprite']

objects_a_glass_of = ['milk', 'juice', 'coffee', 'hot chocolate', 'whisky', 'rum', 'vodka', 'cider', 'lemonade', 'tea', 'water', 'beer',
                                          'coke', 'sprite', 'wine', 'sake']

objects_a_bottle_of = ['kleenex', 'milk', 'juice', 'whisky', 'rum', 'vodka', 'cider', 'lemonade',
                                           'iced tea', 'water', 'beer', 'coke', 'sprite', 'wine','sake']

objects = list(set(objects_a + objects_the + objects_some + objects_an + objects_a_piece_of + objects_a_cup_of + objects_a_can_of + objects_a_bottle_of + objects_a_glass_of))
# ================================================================================================================
# locations
# ================================================================================================================
locations_on = ['nightstand', 'bookshelf', 'coffee table', 'side table', 'kitchen table', 'kitchen cabinet',
                                'tv stand', 'sofa', 'couch', 'bedroom chair', 'kitchen chair', 'living room table', 'center table',
                                'drawer', 'desk', 'cupboard', 'side shelf', 'bookcase', 'dining table', 'fridge', 'counter',
                                'cabinet', 'table', 'bedchamber', 'chair', 'dryer', 'oven', 'rocking chair', 'stove', 'television', 'bed', 'dressing table',
                                'bench', 'futon', 'beanbag', 'stool', 'sideboard', 'washing machine', 'dishwasher']

locations_in = ['wardrobe', 'nightstand', 'bookshelf', 'dining room', 'bedroom', 'closet', 'living room', 'bar', 'office',
                                'drawer', 'kitchen', 'cupboard', 'side shelf', 'refrigerator', 'corridor', 'cabinet', 'bathroom', 'toilet', 'hall', 'hallway',
                                'master bedroom', 'dormitory room', 'bedchamber', 'cellar', 'den', 'garage', 'playroom', 'porch', 'staircase', 'sunroom', 'music room',
                                'prayer room', 'utility room', 'shed', 'basement', 'workshop', 'ballroom', 'box room', 'conservatory', 'drawing room',
                                'games room', 'larder', 'library', 'parlour', 'guestroom', 'crib', 'shower']

locations_at = ['wardrobe', 'nightstand', 'bookshelf', 'coffee table', 'side table', 'kitchen table', 'kitchen cabinet',
                                'bed', 'bedside', 'closet', 'tv stand', 'sofa', 'couch', 'bedroom chair', 'kitchen chair',
                                'living room table', 'center table', 'bar', 'drawer', 'desk', 'cupboard', 'sink', 'side shelf',
                                'bookcase', 'dining table', 'refrigerator', 'counter', 'door', 'cabinet', 'table', 'master bedroom', 'dormitory room',
                                'bedchamber', 'chair', 'dryer', 'entrance', 'garden', 'oven', 'rocking chair', 'room', 'stove', 'television', 'washer',
                                'cellar', 'den', 'laundry', 'pantry', 'patio', 'balcony', 'lamp', 'window', 'lawn', 'cloakroom', 'telephone', 'dressing table',
                                'bench', 'futon', 'radiator', 'washing machine', 'dishwasher']

locations = list(set(locations_at+locations_in+locations_on))

# ================================================================================================================
# names
# ================================================================================================================
names_female = ['hanna', 'barbara', 'samantha', 'erika', 'sophie', 'jackie', 'skyler', 'jane', 'olivia', 'emily', 'amelia', 'lily',
                                'grace', 'ella', 'scarlett', 'isabelle', 'charlotte', 'daisy', 'sienna', 'chloe', 'alice', 'lucy', 'florence', 'rosie',
                                'amelie', 'eleanor', 'emilia', 'amber', 'ivy', 'brooke', 'summer', 'emma', 'rose', 'martha', 'faith', 'amy', 'katie',
                                'madison', 'sarah', 'zoe', 'paige', 'mia', 'emily', 'sophia', 'abigail', 'isabella', 'ava', 'argentina']

names_male = ['ken', 'erick', 'samuel', 'skyler', 'brian', 'thomas', 'edward', 'michael', 'charlie', 'alex', 'john', 'james', 'oscar',
                          'peter', 'oliver', 'jack', 'harry', 'henry', 'jacob', 'thomas', 'william', 'will', 'joshua', 'josh', 'noah', 'ethan', 'joseph',
                          'samuel', 'daniel', 'max', 'logan', 'isaac', 'dylan', 'freddie', 'tyler', 'harrison', 'adam', 'theo', 'arthur', 'toby', 'luke',
                          'lewis', 'matthew', 'harvey', 'ryan', 'tommy', 'michael', 'nathan', 'blake', 'charles', 'connor', 'jamie', 'elliot', 'louis',
                          'aaron', 'evan', 'seth', 'liam', 'mason', 'alexander', 'madison', 'patrick', 'roberto', 'rohan']

pronouns = ['me','us', 'him', 'her', 'them']

names = list(set(names_male+names_female))

# ================================================================================================================
# what to tell
# ================================================================================================================
what_to_tell_about = ['name', 'nationality', 'eye color', 'hair color', 'surname', 'middle name', 'gender', 'pose', 'age', 'job', 'shirt color',
                                           'height', 'mood']

what_to_tell_to = ["your teams affiliation", "your teams country", "your teams name", 'the day of the month','what day is today', 'what day is tomorrow', 'the time',
                                       'the weather', 'that i am coming', 'to wait a moment', 'to come here', 'what time is it', 'a joke', 'something about yourself',
                                       'the name of the person']

# ================================================================================================================
# intros
# ================================================================================================================
intros = ['robot', 'please', 'could you please', 'robot please', 'robot could you please', 'can you', 'robot can you',  'could you', 'robot could you']


# Prining number of objects used in the generator
print('obect_a', len(objects_a))
print('obect_an', len(objects_an))
print('objects_the', len(objects_the))
print('objects_some', len(objects_some))
print('objects_a_piece_of', len(objects_a_piece_of))
print('objects_a_cup_of', len(objects_a_cup_of))
print('objects_a_can_of', len(objects_a_can_of))
print('objects_a_glass_of', len(objects_a_glass_of))
print('objects_a_bottle_of', len(objects_a_bottle_of))
print('objects', len(objects))
print('locations', len(sorted(locations, key=str.lower)))
print('names_female', len(names_female))
print('names_male', len(names_male))
print('what_to_tell_about', len(what_to_tell_about))
print('what_to_tell_to', len(what_to_tell_to))
print('intros', len(intros))
print('-----------------------------------------------------')

# initiating lists (2 per intent)
tasks_take = []; tasks_take_ = []
tasks_follow = []; tasks_follow_ = []
tasks_answer = []; tasks_answer_ = []
tasks_find = []; tasks_find_ = []
tasks_guide = []; tasks_guide_ = []
tasks_tell = []; tasks_tell_ = []
tasks_go = []; tasks_go_ = []
tasks_meet = []; tasks_meet_ = []

#------------------------------------------GO----------------------------------------------
tasks_go_.append(['go - go'])
tasks_go_.append(['navigate - go'])
tasks_go_.append(['proceed - go'])
tasks_go_.append(['move - go'])
tasks_go_.append(['advance - go'])
tasks_go_.append(['travel - go'])
tasks_go_.append(['drive - go'])
tasks_go_.append(['come - go'])
tasks_go_.append(['go near - go'])
tasks_go_.append(['walk - go'])
tasks_go_.append(['reach - go'])

tasks_go_.append(['go to ' + name + ' - go' for name in names])
tasks_go_.append(['navigate to ' + name + ' - go' for name in names])
tasks_go_.append(['proceed to ' + name + ' - go' for name in names])
tasks_go_.append(['move to ' + name + ' - go' for name in names])
tasks_go_.append(['advance to ' + name + ' - go' for name in names])
tasks_go_.append(['travel to ' + name + ' - go' for name in names])
tasks_go_.append(['drive to ' + name + ' - go' for name in names])
tasks_go_.append(['come to ' + name + ' - go' for name in names])
tasks_go_.append(['go near to ' + name + ' - go' for name in names])
tasks_go_.append(['walk to ' + name + ' - go' for name in names])
tasks_go_.append(['reach ' + name + ' - go' for name in names])

tasks_go_.append(['go to the ' + location + ' - go' for location in locations])
tasks_go_.append(['navigate to the ' + location + ' - go' for location in locations])
tasks_go_.append(['proceed to the ' + location + ' - go' for location in locations])
tasks_go_.append(['move to the ' + location + ' - go' for location in locations])
tasks_go_.append(['advance to the ' + location + ' - go' for location in locations])
tasks_go_.append(['travel to the ' + location + ' - go' for location in locations])
tasks_go_.append(['drive to the ' + location + ' - go' for location in locations])
tasks_go_.append(['come to the ' + location + ' - go' for location in locations])
tasks_go_.append(['go near the ' + location + ' - go' for location in locations])
tasks_go_.append(['walk to the ' + location + ' - go' for location in locations])
tasks_go_.append(['reach the ' + location + ' - go' for location in locations])

tasks_go_.append(['go to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['navigate to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['proceed to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['move to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['advance to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['travel to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['drive to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['come to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['go near ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['walk to ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])
tasks_go_.append(['reach ' + name + ' at the ' + location + ' - go' for name in names for location in locations_at])

tasks_go_.append(['go to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['navigate to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['proceed to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['move to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['advance to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['travel to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['drive to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['come to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['go near ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['walk to ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])
tasks_go_.append(['reach ' + name + ' in the ' + location + ' - go' for name in names for location in locations_in])

# resampling and appending individual structures
len_of_str = [len(tasks_go_[i]) for i in range(len(tasks_go_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider

for i in range(len(tasks_go_)):
    # resample if len not enough or more
     if len_of_str[i]!=mean_of_strct_lens:
         try: tasks_go_[i] = resample(tasks_go_[i], n_samples=mean_of_strct_lens, replace=False)
         except:  tasks_go_[i] = resample(tasks_go_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_go = [item for sublist in tasks_go_ for item in sublist]
# rem temp params
del tasks_go_, len_of_str, mean_of_strct_lens
print ("number of 'go' sentences", len(tasks_go))

#----------------------------------------TAKE---------------------------------------------
tasks_take_.append(['grasp the ' + object + ' - take' for object in objects])
tasks_take_.append(['pick up the ' + object + ' - take' for object in objects])

tasks_take_.append(['bring me the ' + object + ' - take' for object in objects])
tasks_take_.append(['give me the ' + object + ' - take' for object in objects])

tasks_take_.append(['take the ' + object + ' to the ' + location + ' - take' for object in objects for location in locations])
tasks_take_.append(['put the ' + object + ' to the ' + location + ' - take' for object in objects for location in locations])
tasks_take_.append(['deliver the ' + object + ' to the ' + location + ' - take' for object in objects for location in locations])

tasks_take_.append(['take the ' + object + ' to ' + name + ' - take' for object in objects for name in names])
tasks_take_.append(['deliver the ' + object + ' to ' + name + ' - take' for object in objects for name in names])
tasks_take_.append(['give the ' + object + ' to ' + name + ' - take' for object in objects for name in names])

tasks_take_.append(['grasp the ' + object + ' from the ' + location + ' - take' for object in objects for location in locations])
tasks_take_.append(['pick up the ' + object + ' from the ' + location + ' - take' for object in objects for location in locations])

tasks_take_.append(['bring the ' + object + ' to ' + name + ' - take' for object in objects_the for name in names])

tasks_take_.append(['bring me the ' + object + ' from the ' + location + ' - take' for object in objects_the for location in locations])
tasks_take_.append(['give me the ' + object + ' from the ' + location + ' - take' for object in objects_the for location in locations])

tasks_take_.append(['bring the ' + object + ' to ' + name + ' at the ' + location + ' - take' for object in objects_the for name in names for location in locations])

tasks_take_.append(['bring the ' + object + ' to me - take' for object in objects_the])
tasks_take_.append(['deliver the ' + object + ' to me - take' for object in objects_the])
tasks_take_.append(['give the ' + object + ' to me - take' for object in objects_the])

tasks_take_.append(['deliver the ' + object + ' to ' + name + ' at the ' + location + ' - take' for object in objects_the for name in names for location in locations_at])
tasks_take_.append(['give the ' + object + ' to ' + name + ' at the ' + location + ' - take' for object in objects_the for name in names for location in locations_at])

tasks_take_.append(['bring to ' + name + ' at the ' + location + ' the ' + object + ' from the ' + location2 + ' - take' for name in names for location in locations_at[:int(len(locations_at)/2)] for object in objects_the for location2 in locations[:int(len(locations)/4)] if location!=location2])
tasks_take_.append(['give to ' + name + ' at the ' + location + ' the ' + object + ' from the ' + location2 + ' - take' for name in names for location in locations_at[:int(len(locations_at)/2)] for object in objects_the for location2 in locations[:int(len(locations)/4)] if location!=location2])

tasks_take_.append(['deliver the ' + object + ' to ' + name + ' - take' for object in objects_the for name in names])

tasks_take_.append(['get the ' + object + ' from the ' + location + ' - take' for object in objects_the for location in locations])
tasks_take_.append(['get the ' + object + ' to the ' + location + ' - take' for object in objects_the for location in locations])
tasks_take_.append(['take the ' + object + ' from the ' + location + ' - take' for object in objects_the for location in locations])

tasks_take_.append(['get the ' + object + ' from the ' + location + ' to the ' + location2 + ' - take' for object in objects_the for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])
tasks_take_.append(['take the ' + object + ' from the ' + location + ' to the ' + location2 + ' - take' for object in objects_the for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])

tasks_take_.append(['place the ' + object + ' on the ' + location + ' - take' for object in objects_the for location in locations_on])
tasks_take_.append(['put the ' + object + ' on the ' + location + ' - take' for object in objects_the for location in locations_on])

tasks_take_.append(['grasp the ' + object + ' to the ' + location + ' - take' for object in objects for location in locations])
tasks_take_.append(['pick up the ' + object + ' to the ' + location + ' - take' for object in objects for location in locations])

tasks_take_.append(['grasp the ' + object + ' from the ' + location + ' to the ' + location2 + ' - take' for object in objects for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])
tasks_take_.append(['pick up the ' + object + ' from the ' + location + ' to the ' + location2 + ' - take' for object in objects for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])

# resampling and appending individual structures
len_of_str = [len(tasks_take_[i]) for i in range(len(tasks_take_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_take_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_take_[i] = resample(tasks_take_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_take_[i] = resample(tasks_take_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_take = [item for sublist in tasks_take_ for item in sublist]
# rem temp params
del tasks_take_, len_of_str, mean_of_strct_lens
print ("number of 'take' sentences", len(tasks_take))

#-----------------------------------------------FIND-----------------------------------------------
tasks_find_.append(['find the ' + object + ' - find' for object in objects])
tasks_find_.append(['look for the ' + object + ' - find' for object in objects])
tasks_find_.append(['locate the ' + object + ' - find' for object in objects])
tasks_find_.append(['pinpoint the ' + object + ' - find' for object in objects])
tasks_find_.append(['spot the ' + object + ' - find' for object in objects])

tasks_find_.append(['find the ' + object + ' in the ' + location + ' - find' for object in objects for location in locations_in])
tasks_find_.append(['look for the ' + object + ' in the ' + location + ' - find' for object in objects for location in locations_in])
tasks_find_.append(['locate the ' + object + ' in the ' + location + ' - find' for object in objects for location in locations_in])
tasks_find_.append(['pinpoint the ' + object + ' in the ' + location + ' - find' for object in objects for location in locations_in])
tasks_find_.append(['spot the ' + object + ' in the ' + location + ' - find' for object in objects for location in locations_in])

tasks_find_.append(['find ' + name + ' - find' for name in names])
tasks_find_.append(['look for ' + name + ' - find' for name in names])
tasks_find_.append(['locate ' + name + ' - find' for name in names])
tasks_find_.append(['pinpoint ' + name + ' - find' for name in names])
tasks_find_.append(['spot ' + name + ' - find' for name in names])

tasks_find_.append(['find ' + name + ' in the ' + location + ' - find' for name in names for location in locations_in])
tasks_find_.append(['look for ' + name + ' in the ' + location + ' - find' for name in names for location in locations_in])
tasks_find_.append(['locate ' + name + ' in the ' + location + ' - find' for name in names for location in locations_in])
tasks_find_.append(['pinpoint ' + name + ' in the ' + location + ' - find' for name in names for location in locations_in])
tasks_find_.append(['spot ' + name + ' in the ' + location + ' - find' for name in names for location in locations_in])

tasks_find_.append(['find ' + name + ' at the ' + location + ' - find' for name in names for location in locations_at])
tasks_find_.append(['look for ' + name + ' at the ' + location + ' - find' for name in names for location in locations_at])
tasks_find_.append(['locate ' + name + ' at the ' + location + ' - find' for name in names for location in locations_at])
tasks_find_.append(['pinpoint ' + name + ' at the ' + location + ' - find' for name in names for location in locations_at])
tasks_find_.append(['spot ' + name + ' at the ' + location + ' - find' for name in names for location in locations_at])

tasks_find_.append(['find someone - find'])
tasks_find_.append(['locate someone - find'])
tasks_find_.append(['look for someone - find'])
tasks_find_.append(['find a person - find'])
tasks_find_.append(['locate a person - find'])
tasks_find_.append(['look for a person - find'])

tasks_find_.append(['find a person' + ' in the ' + location + ' - find' for location in locations_in])
tasks_find_.append(['locate a person' + ' in the ' + location + ' - find' for location in locations_in])
tasks_find_.append(['look for a person' + ' in the ' + location + ' - find' for location in locations_in])
tasks_find_.append(['find someone' + ' in the ' + location + ' - find' for location in locations_in])
tasks_find_.append(['look for someone' + ' in the ' + location + ' - find' for location in locations_in])
tasks_find_.append(['locate someone' + ' in the ' + location + ' - find' for location in locations_in])

# resampling and appending individual structures
len_of_str = [len(tasks_find_[i]) for i in range(len(tasks_find_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_find_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_find_[i] = resample(tasks_find_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_find_[i] = resample(tasks_find_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_find = [item for sublist in tasks_find_ for item in sublist]
# rem temp params
del tasks_find_, len_of_str, mean_of_strct_lens
print ("number of 'find' sentences", len(tasks_find))

#--------------------------------------------ANSWER-------------------------------------
tasks_answer_.append(['answer a question - answer'])
tasks_answer_.append(['answer a question to ' + name + ' - answer' for name in names])
tasks_answer_.append(['answer a question to ' + name + ' at the ' + location + ' - answer' for name in names for location in locations])
tasks_answer_.append(['answer a question to ' + name + ' in the ' + location + ' - answer' for name in names for location in locations_in])

# resampling and appending individual structures
len_of_str = [len(tasks_answer_[i]) for i in range(len(tasks_answer_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_answer_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_answer_[i] = resample(tasks_answer_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_answer_[i] = resample(tasks_answer_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_answer = [item for sublist in tasks_answer_ for item in sublist]
# rem temp params
del tasks_answer_, len_of_str, mean_of_strct_lens
print ("number of 'answer' sentences", len(tasks_answer))

#---------------------------------------------TELL-------------------------------------
tasks_tell_.append(['tell ' + w + ' to ' + name + ' - tell' for w in what_to_tell_to for name in names])
tasks_tell_.append(['say ' + w + ' to ' + name + ' - tell' for w in what_to_tell_to for name in names])
tasks_tell_.append(['tell ' + w + ' to ' + name + ' at the ' + location + ' - tell' for w in what_to_tell_to for name in names for location in locations])
tasks_tell_.append(['say ' + w + ' to ' + name + ' at the ' + location + ' - tell' for w in what_to_tell_to for name in names for location in locations])
tasks_tell_.append(['tell ' + w + ' to ' + name + ' in the ' + location + ' - tell' for w in what_to_tell_to for name in names for location in locations_in])
tasks_tell_.append(['say ' + w + ' to ' + name + ' in the ' + location + ' - tell' for w in what_to_tell_to for name in names for location in locations_in])

tasks_tell_.append(['say ' + w + ' - tell' for w in what_to_tell_to])
tasks_tell_.append(['tell ' + w + ' - tell' for w in what_to_tell_to])
tasks_tell_.append(['tell me ' + w + ' - tell' for w in what_to_tell_to])

tasks_tell_.append(['tell me the name of the person at the ' + location + ' - tell' for location in locations_at])
tasks_tell_.append(['tell me the name of the person in the ' + location + ' - tell' for location in locations_in])
tasks_tell_.append(['tell me how many ' + object + ' there are on the ' + location + ' - tell' for object in objects for location in locations_on])

# resampling and appending individual structures
len_of_str = [len(tasks_tell_[i]) for i in range(len(tasks_tell_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_tell_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_tell_[i] = resample(tasks_tell_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_tell_[i] = resample(tasks_tell_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_tell = [item for sublist in tasks_tell_ for item in sublist]
# rem temp params
del tasks_tell_, len_of_str, mean_of_strct_lens
print ("number of 'tell' sentences", len(tasks_tell))

#---------------------------------------------GUIDE-------------------------------------
tasks_guide_.append(['accompany ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['conduct ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['escort ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['guide ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['lead ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['take ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['oversee ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['supervise ' + pronoun + ' - guide' for pronoun in pronouns])
tasks_guide_.append(['usher ' + pronoun + ' - guide' for pronoun in pronouns])

tasks_guide_.append(['accompany ' + name + ' - guide' for name in names])
tasks_guide_.append(['conduct ' + name + ' - guide' for name in names])
tasks_guide_.append(['escort ' + name + ' - guide' for name in names])
tasks_guide_.append(['guide ' + name + ' - guide' for name in names])
tasks_guide_.append(['lead ' + name + ' - guide' for name in names])
tasks_guide_.append(['take ' + name + ' - guide' for name in names])
tasks_guide_.append(['oversee ' + name + ' - guide' for name in names])
tasks_guide_.append(['supervise ' + name + ' - guide' for name in names])
tasks_guide_.append(['usher ' + name + ' - guide' for name in names])

tasks_guide_.append(['accompany ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['conduct ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['escort ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['guide ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['lead ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['take ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['oversee ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['supervise ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])
tasks_guide_.append(['usher ' + pronoun + ' to the ' + location + ' - guide' for pronoun in pronouns for location in locations])

tasks_guide_.append(['accompany ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['conduct ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['escort ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['guide ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['lead ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['take ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['oversee ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['supervise ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])
tasks_guide_.append(['usher ' + name + ' to the ' + location + ' - guide' for name in names for location in locations])

# resampling and appending individual structures
len_of_str = [len(tasks_guide_[i]) for i in range(len(tasks_guide_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_guide_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_guide_[i] = resample(tasks_guide_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_guide_[i] = resample(tasks_guide_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_guide = [item for sublist in tasks_guide_ for item in sublist]
# rem temp params
del tasks_guide_, len_of_str, mean_of_strct_lens
print ("number of 'guide' sentences", len(tasks_guide))

#---------------------------------------------FOLLOW-------------------------------------
tasks_follow_.append(['come after ' + pronoun + ' - follow' for pronoun in pronouns])
tasks_follow_.append(['go after ' + pronoun + ' - follow' for pronoun in pronouns])
tasks_follow_.append(['come behind ' + pronoun + ' - follow' for pronoun in pronouns])
tasks_follow_.append(['go behind ' + pronoun + ' - follow' for pronoun in pronouns])
tasks_follow_.append(['follow ' + pronoun + ' - follow' for pronoun in pronouns])
tasks_follow_.append(['pursue ' + pronoun + ' - follow' for pronoun in pronouns])
tasks_follow_.append(['chase ' + pronoun + ' - follow' for pronoun in pronouns])

tasks_follow_.append(['come after ' + name + ' - follow' for name in names])
tasks_follow_.append(['go after ' + name + ' - follow' for name in names])
tasks_follow_.append(['come behind ' + name + ' - follow' for name in names])
tasks_follow_.append(['go behind ' + name + ' - follow' for name in names])
tasks_follow_.append(['follow ' + name + ' - follow' for name in names])
tasks_follow_.append(['pursue ' + name + ' - follow' for name in names])
tasks_follow_.append(['chase ' + name + ' - follow' for name in names])

tasks_follow_.append(['come after '  + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])
tasks_follow_.append(['go after ' + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])
tasks_follow_.append(['come behind ' + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])
tasks_follow_.append(['go behind ' + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])
tasks_follow_.append(['follow ' + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])
tasks_follow_.append(['pursue ' + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])
tasks_follow_.append(['chase ' + pronoun + ' to the ' + location + ' - follow' for pronoun in pronouns for location in locations])

tasks_follow_.append(['come after '  + name + ' to the ' + location + ' - follow' for name in names for location in locations])
tasks_follow_.append(['go after ' + name + ' to the ' + location + ' - follow' for name in names for location in locations])
tasks_follow_.append(['come behind ' + name + ' to the ' + location + ' - follow' for name in names for location in locations])
tasks_follow_.append(['go behind ' + name + ' to the ' + location + ' - follow' for name in names for location in locations])
tasks_follow_.append(['follow ' + name + ' to the ' + location + ' - follow' for name in names for location in locations])
tasks_follow_.append(['pursue ' + name + ' to the ' + location + ' - follow' for name in names for location in locations])
tasks_follow_.append(['chase ' + name + ' to the ' + location + ' - follow' for name in names for location in locations])

# resampling and appending individual structures
len_of_str = [len(tasks_follow_[i]) for i in range(len(tasks_follow_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_follow_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_follow_[i] = resample(tasks_follow_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_follow_[i] = resample(tasks_follow_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_follow = [item for sublist in tasks_follow_ for item in sublist]
# rem temp params
del tasks_follow_, len_of_str, mean_of_strct_lens
print ("number of 'follow' sentences", len(tasks_follow))

#---------------------------------------------meet-------------------------------------
tasks_meet_.append(['meet ' + pronoun + ' - meet' for pronoun in pronouns])
tasks_meet_.append(['encounter ' + pronoun + ' - meet' for pronoun in pronouns])
tasks_meet_.append(['face ' + pronoun + ' - meet' for pronoun in pronouns])
tasks_meet_.append(['greet ' + pronoun + ' - meet' for pronoun in pronouns])
tasks_meet_.append(['see ' + pronoun + ' - meet' for pronoun in pronouns])
tasks_meet_.append(['stumble ' + pronoun + ' - meet' for pronoun in pronouns])
tasks_meet_.append(['salute ' + pronoun + ' - meet' for pronoun in pronouns])

tasks_meet_.append(['meet ' + name + ' - meet' for name in names])
tasks_meet_.append(['encounter ' + name + ' - meet' for name in names])
tasks_meet_.append(['face ' + name + ' - meet' for name in names])
tasks_meet_.append(['greet ' + name + ' - meet' for name in names])
tasks_meet_.append(['see ' + name + ' - meet' for name in names])
tasks_meet_.append(['stumble ' + name + ' - meet' for name in names])
tasks_meet_.append(['salute ' + name + ' - meet' for name in names])

tasks_meet_.append(['meet '  + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])
tasks_meet_.append(['encounter ' + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])
tasks_meet_.append(['face ' + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])
tasks_meet_.append(['greet ' + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])
tasks_meet_.append(['see ' + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])
tasks_meet_.append(['stumble ' + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])
tasks_meet_.append(['salute ' + pronoun + ' at the ' + location + ' - meet' for pronoun in pronouns for location in locations])

tasks_meet_.append(['meet '  + name + ' at the ' + location + ' - meet' for name in names for location in locations])
tasks_meet_.append(['encounter ' + name + ' at the ' + location + ' - meet' for name in names for location in locations])
tasks_meet_.append(['face ' + name + ' at the ' + location + ' - meet' for name in names for location in locations])
tasks_meet_.append(['greet ' + name + ' at the ' + location + ' - meet' for name in names for location in locations])
tasks_meet_.append(['see ' + name + ' at the ' + location + ' - meet' for name in names for location in locations])
tasks_meet_.append(['stumble ' + name + ' at the ' + location + ' - meet' for name in names for location in locations])
tasks_meet_.append(['salute ' + name + ' at the ' + location + ' - meet' for name in names for location in locations])

# resampling and appending individual structures
len_of_str = [len(tasks_meet_[i]) for i in range(len(tasks_meet_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_meet_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_meet_[i] = resample(tasks_meet_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_meet_[i] = resample(tasks_meet_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_meet = [item for sublist in tasks_meet_ for item in sublist]
# rem temp params
del tasks_meet_, len_of_str, mean_of_strct_lens
print ("number of 'meet' sentences", len(tasks_meet))
print('-----------------------------------------------------')

#----------------------------------------------------------------------------------------------
print('resampling and appending all the task sentences into one list')
tasks = []
if len(tasks_go)>1:
    # resample
    try: tasks_go = resample(tasks_go, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_go = resample(tasks_go, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_go[i])
    # rm temp params
    del tasks_go
if len(tasks_take)>1:
    # resample
    try: tasks_take = resample(tasks_take, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_take = resample(tasks_take, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_take[i])
    # rm temp params
    del tasks_take
if len(tasks_find)>1:
    # resample
    try: tasks_find = resample(tasks_find, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_find = resample(tasks_find, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_find[i])
    # rm temp params
    del tasks_find
if len(tasks_answer)>1:
    # resample
    try: tasks_answer = resample(tasks_answer, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_answer = resample(tasks_answer, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_answer[i])
    # rm temp params
    del tasks_answer
if len(tasks_tell)>1:
    # resample
    try: tasks_tell = resample(tasks_tell, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_tell = resample(tasks_tell, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_tell[i])
    # rm temp params
    del tasks_tell
if len(tasks_meet)>1:
    # resample
    try: tasks_meet = resample(tasks_meet, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_meet = resample(tasks_meet, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_meet[i])
    # rm temp params
    del tasks_meet
if len(tasks_follow)>1:
    # resample
    try: tasks_follow = resample(tasks_follow, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_follow = resample(tasks_follow, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_follow[i])
    # rm temp params
    del tasks_follow
if len(tasks_guide)>1:
    # resample
    try: tasks_guide = resample(tasks_guide, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_guide = resample(tasks_guide, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_guide[i])
    # rm temp params
    del tasks_guide

print('-----------------------------------------------------')
print("total number of sentences ", len(tasks))

# splitting inputs and labels and adding intros
h = 0
sentences = []
outputs = []
for v in range(len(tasks)):

    try:
        task = tasks[v].split('- ')
    except:
        print('error at {}, check if the items in tasks are strings tasks[v] = {}'.format(v, tasks[v]))
        break
    sentence = task[0]
    output = task[1]

    # removing sentences with more than 15 length
    if len(sentence.split())>15: continue

    # adding intro [hello, hi etc]
    if v%4 == 0:
        # Take out the sentences which are longer than 15 words (The number is choosen by Pedro Martins, ref: mbot_nlu/ros/doc/pedro_thesis.pdf)
        # If the combined lenth of the introduction and the sentence is greater than 15, the introduction is not added.
        if (len(intros[h].split()) + len(sentence.split()))<=15:
            sentence = intros[h]+ ' ' + sentence
            h = h + 1
            # Resetting the intros index if it reaches the last item in the list
            if h == len(intros): h = 0
        else:
            pass

    if sentence[-1] == ' ': sentence = sentence[:-1]
    sentences.append(sentence)
    outputs.append(output)

print("Size of sentence list ", len(sentences))
print("First output: ", sentences[0])
print("Size of output list ", len(outputs))
print("First output: ", outputs[0])


# Dumping the serialized inputs and outputs using pickle
with open('inputs', 'wb') as inputs_file:
    msgpack.dump(sentences, inputs_file)
with open('outputs', 'wb') as outputs_file:
    msgpack.dump(outputs, outputs_file)

# Take out the sentences which are longer than 15 words (The number is choosen by Pedro Martins, ref: mbot_nlu/ros/doc/pedro_thesis.pdf)
# 15 is the number of words sampled as it was recomended in Mikolov 2013
matches = []
matches_lens = 0
for sentence in sentences:
    sentence_split = sentence.split()
    if ' ' in sentence_split:
        sentence_split = sentence_split.remove(' ')
    if len(sentence_split)>15:
        matches.append(sentence)
        matches_lens = matches_lens + 1
    else:
        continue

print('-----------------------------------------------------')
print('sentences with more than 15 words :')
# Printing each sentence one row at a time
for match in matches: print(match)
print('-----------------------------------------------------')
print('number of sentences with more than 15 words', matches_lens)
print('Total number of inputs', len(sentences))
print('Total number of outputs', len(outputs))
print('-----------------------------------------------------')
print('Data generation is complete for Intent training, you may start the training by running training_nn_model.py script')
