#!/usr/bin/env python3

import random
import numpy as np
from sklearn.utils import resample

# number of types of different structured sentences in each of intent classes
n_struct = {'go': 66, 'take': 46, 'find': 62, 'answer': 7, 'tell': 17, 'guide': 36, 'follow': 28, 'meet': 28}

data_slider = 0

random_state = None

n_samples_per_intent = int(200000/len(n_struct))

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
                           'glass','pasta','can', 'bottle', 'fork', 'knife', 'bowl', 'tray', 'plate', 'newspaper', 'magazine', 'rice','kleenex', 'whiteboard cleaner', 'cup']

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
people = ['person standing','person lying','person waving', 'people sitting', 'person',
          'people', 'man','woman','boy','girl', 'men','women']

pronouns = ['me', 'us', 'him', 'her', 'them']
pronoun_it = 'it'

names = list(set(names_male+names_female+people))

# ================================================================================================================
# what to tell
# ================================================================================================================
what_to_tell_about = ['name', 'nationality', 'eye color', 'hair color', 'surname', 'middle name', 'gender', 'pose', 'age', 'job', 'shirt color',
                                           'height', 'mood']

what_to_tell_to = ["your teams affiliation", "your teams country", "your teams name", 'the day of the month','what day is today', 'what day is tomorrow', 'the time',
                                       'the weather', 'that i am coming', 'to wait a moment', 'to come here', 'what time is it', 'a joke', 'something about yourself',
                                       'name of the person', "what's the largest", "what's the thinnest", "what's the biggest"]

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

tasks_go_.append(['go to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['navigate to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['proceed to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['move to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['advance to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['travel to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['drive to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['come to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['go near to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['walk to ' + '[' + name + '](person)' for name in names])
tasks_go_.append(['reach ' + '[' + name + '](person)' for name in names])

tasks_go_.append(['go to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['navigate to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['proceed to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['move to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['advance to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['travel to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['drive to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['come to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['go near to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['walk to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_go_.append(['reach ' + '[' + pronoun + '](person)' for pronoun in pronouns])

tasks_go_.append(['go to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['navigate to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['proceed to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['move to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['advance to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['travel to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['drive to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['come to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['go near to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['walk to ' + '[' + pronoun_it + '](object)'])
tasks_go_.append(['reach ' + '[' + pronoun_it + '](object)'])

tasks_go_.append(['go to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['navigate to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['proceed to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['move to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['advance to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['travel to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['drive to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['come to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['go near the ' + '[' + location + ']' for location in locations])
tasks_go_.append(['walk to the ' + '[' + location + '](destination)' for location in locations])
tasks_go_.append(['reach the ' + '[' + location + ']' for location in locations])

tasks_go_.append(['go to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['navigate to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['proceed to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['move to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['advance to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['travel to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['drive to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['come to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['go near ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['walk to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_go_.append(['reach ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])

tasks_go_.append(['go to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['navigate to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['proceed to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['move to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['advance to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['travel to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['drive to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['come to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['go near ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['walk to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_go_.append(['reach ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])

# resampling and appending individual structures
len_of_str = [len(tasks_go_[i]) for i in range(len(tasks_go_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider

for i in range(len(tasks_go_)):
    # resample if len not enough or more
     if len_of_str[i] != mean_of_strct_lens:
         try: tasks_go_[i] = resample(tasks_go_[i], n_samples=mean_of_strct_lens, replace=False)
         except:  tasks_go_[i] = resample(tasks_go_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_go = [item for sublist in tasks_go_ for item in sublist]
# rem temp params
del tasks_go_, len_of_str, mean_of_strct_lens
print ("number of 'go' sentences", len(tasks_go))

#----------------------------------------TAKE---------------------------------------------
tasks_take_.append(['grasp ' + '[' + pronoun_it + '](object)'])
tasks_take_.append(['pick ' + '[' + pronoun_it + '](object) up'])

tasks_take_.append(['bring ' + '[' + pronoun_it + '](object)' + ' to ' + '[' + name + '](person)' for name in names])
tasks_take_.append(['give ' + '[' + pronoun_it + '](object)' + ' to ' + '[' + name + '](person)' for name in names])
tasks_take_.append(['deliver ' + '[' + pronoun_it + '](object)' + ' to ' + '[' + name + '](person)' for name in names])

tasks_take_.append(['bring ' + '[' + pronoun_it + '](object)' + ' to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_take_.append(['give ' + '[' + pronoun_it + '](object)' + ' to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_take_.append(['deliver ' + '[' + pronoun_it + '](object)' + ' to ' + '[' + pronoun + '](person)' for pronoun in pronouns])

tasks_take_.append(['take ' + '[' + pronoun_it + '](object)' + ' to the ' + '[' + location + '](destination)' for location in locations])
tasks_take_.append(['deliver ' + '[' + pronoun_it + '](object)' + ' to the ' + '[' + location + '](destination)' for location in locations])

tasks_take_.append(['take ' + '[' + pronoun_it + '](object)' + ' from the ' + '[' + location + '](source)' for location in locations])
tasks_take_.append(['deliver ' + '[' + pronoun_it + '](object)' + ' from the ' + '[' + location + '](source)' for location in locations])

tasks_take_.append(['grasp the ' + '[' + object + '](object)' for object in objects])
tasks_take_.append(['pick up the ' + '[' + object + '](object)' for object in objects])

tasks_take_.append(['bring [' + pronoun + '](person)' + ' the ' + '[' + object + '](object)' for pronoun in pronouns for object in objects])
tasks_take_.append(['give [' + pronoun + '](person)' + ' the ' + '[' + object + '](object)' for pronoun in pronouns for object in objects])

tasks_take_.append(['take the ' + '[' + object + '](object)' + ' to the ' + '[' + location + '](destination)' for object in objects for location in locations])
tasks_take_.append(['deliver the ' + '[' + object + '](object)' + ' to the ' + '[' + location + '](destination)' for object in objects for location in locations])

tasks_take_.append(['take the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' for object in objects for name in names])
tasks_take_.append(['deliver the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' for object in objects for name in names])
tasks_take_.append(['give the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' for object in objects for name in names])

tasks_take_.append(['grasp the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' for object in objects for location in locations])
tasks_take_.append(['pick up the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' for object in objects for location in locations])

tasks_take_.append(['bring the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' for object in objects_the for name in names])

tasks_take_.append(['bring [' + pronoun + '](person)' + ' the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' for pronoun in pronouns for object in objects_the for location in locations])
tasks_take_.append(['give [' + pronoun + '](person)' + ' the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' for pronoun in pronouns for object in objects_the for location in locations])

tasks_take_.append(['bring the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for object in objects_the for name in names for location in locations])

tasks_take_.append(['bring the ' + '[' + object + '](object)' + ' to [' + pronoun + '](person)' for object in objects_the for pronoun in pronouns])
tasks_take_.append(['deliver the ' + '[' + object + '](object)' + ' to [' + pronoun + '](person)' for object in objects_the for pronoun in pronouns])
tasks_take_.append(['give the ' + '[' + object + '](object)' + ' to [' + pronoun + '](person)' for object in objects_the for pronoun in pronouns])

tasks_take_.append(['deliver the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for object in objects_the for name in names for location in locations_at])
tasks_take_.append(['give the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for object in objects_the for name in names for location in locations_at])

tasks_take_.append(['bring to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' + ' the ' + '[' + object + '](object)' + ' from the ' + '[' + location2 + '](source)' for name in names for location in locations_at[:int(len(locations_at)/2)] for object in objects_the for location2 in locations[:int(len(locations)/4)] if location!=location2])
tasks_take_.append(['give to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' + ' the ' + '[' + object + '](object)' + ' from the ' + '[' + location2 + '](source)' for name in names for location in locations_at[:int(len(locations_at)/2)] for object in objects_the for location2 in locations[:int(len(locations)/4)] if location!=location2])

tasks_take_.append(['deliver the ' + '[' + object + '](object)' + ' to ' + '[' + name + '](person)' for object in objects_the for name in names])

tasks_take_.append(['get the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' for object in objects_the for location in locations])
tasks_take_.append(['get the ' + '[' + object + '](object)' + ' to the ' + '[' + location + '](destination)' for object in objects_the for location in locations])
tasks_take_.append(['take the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' for object in objects_the for location in locations])

tasks_take_.append(['get the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' + ' to the ' + '[' + location2 + '](destination)' for object in objects_the for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])
tasks_take_.append(['take the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' + ' to the ' + '[' + location2 + '](destination)' for object in objects_the for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])

tasks_take_.append(['place the ' + '[' + object + '](object)' + ' on the ' + '[' + location + '](destination)' for object in objects_the for location in locations_on])
tasks_take_.append(['put the ' + '[' + object + '](object)' + ' on the ' + '[' + location + '](destination)' for object in objects_the for location in locations_on])

tasks_take_.append(['grasp the ' + '[' + object + '](object)' + ' to the ' + '[' + location + '](destination)' for object in objects for location in locations])
tasks_take_.append(['pick up the ' + '[' + object + '](object)' + ' to the ' + '[' + location + '](destination)' for object in objects for location in locations])

tasks_take_.append(['grasp the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' + ' to the ' + '[' + location2 + '](destination)' for object in objects for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])
tasks_take_.append(['pick up the ' + '[' + object + '](object)' + ' from the ' + '[' + location + '](source)' + ' to the ' + '[' + location2 + '](destination)' for object in objects for location in locations[:int(len(locations)/4)] for location2 in locations[:int(len(locations)/4)] if location!=location2])

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
tasks_find_.append(['find the ' + '[' + object + '](object)' for object in objects])
tasks_find_.append(['look for the ' + '[' + object + '](object)' for object in objects])
tasks_find_.append(['locate the ' + '[' + object + '](object)' for object in objects])
tasks_find_.append(['pinpoint the ' + '[' + object + '](object)' for object in objects])
tasks_find_.append(['spot the ' + '[' + object + '](object)' for object in objects])

tasks_find_.append(['find the ' + '[' + pronoun_it + '](object)'])
tasks_find_.append(['look for the ' + '[' + pronoun_it + '](object)'])
tasks_find_.append(['locate the ' + '[' + pronoun_it + '](object)'])
tasks_find_.append(['pinpoint the ' + '[' + pronoun_it + '](object)'])
tasks_find_.append(['spot the ' + '[' + pronoun_it + '](object)'])

tasks_find_.append(['find the ' + '[' + object + '](object)' + ' in the ' + '[' + location + '](destination)' for object in objects for location in locations_in])
tasks_find_.append(['look for the ' + '[' + object + '](object)' + ' in the ' + '[' + location + '](destination)' for object in objects for location in locations_in])
tasks_find_.append(['locate the ' + '[' + object + '](object)' + ' in the ' + '[' + location + '](destination)' for object in objects for location in locations_in])
tasks_find_.append(['pinpoint the ' + '[' + object + '](object)' + ' in the ' + '[' + location + '](destination)' for object in objects for location in locations_in])
tasks_find_.append(['spot the ' + '[' + object + '](object)' + ' in the ' + '[' + location + '](destination)' for object in objects for location in locations_in])

tasks_find_.append(['find the ' + '[' + pronoun_it + '](object)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['look for the ' + '[' + pronoun_it + '](object)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['locate the ' + '[' + pronoun_it + '](object)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['pinpoint the ' + '[' + pronoun_it + '](object)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['spot the ' + '[' + pronoun_it + '](object)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])


tasks_find_.append(['find ' + '[' + name + '](person)' for name in names])
tasks_find_.append(['look for ' + '[' + name + '](person)' for name in names])
tasks_find_.append(['locate ' + '[' + name + '](person)' for name in names])
tasks_find_.append(['pinpoint ' + '[' + name + '](person)' for name in names])
tasks_find_.append(['spot ' + '[' + name + '](person)' for name in names])

tasks_find_.append(['find ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_find_.append(['look for ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_find_.append(['locate ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_find_.append(['pinpoint ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_find_.append(['spot ' + '[' + pronoun + '](person)' for pronoun in pronouns])

tasks_find_.append(['find ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_find_.append(['look for ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_find_.append(['locate ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_find_.append(['pinpoint ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])
tasks_find_.append(['spot ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])

tasks_find_.append(['find ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_find_.append(['look for ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_find_.append(['locate ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_find_.append(['pinpoint ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])
tasks_find_.append(['spot ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations_at])

tasks_find_.append(['find ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_in])
tasks_find_.append(['look for ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_in])
tasks_find_.append(['locate ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_in])
tasks_find_.append(['pinpoint ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_in])
tasks_find_.append(['spot ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_in])

tasks_find_.append(['find ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_at])
tasks_find_.append(['look for ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_at])
tasks_find_.append(['locate ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_at])
tasks_find_.append(['pinpoint ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_at])
tasks_find_.append(['spot ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_at])

tasks_find_.append(['find ' + '[someone](person)'])
tasks_find_.append(['locate ' + '[someone](person)'])
tasks_find_.append(['look for ' + '[someone](person)'])
tasks_find_.append(['find a '+ '[person](person)'])
tasks_find_.append(['locate a ' + '[person](person)'])
tasks_find_.append(['look for a ' + '[person](person)'])

tasks_find_.append(['find a person' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['locate a person' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['look for a person' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['find ' + '[someone](person)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['look for ' + '[someone](person)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])
tasks_find_.append(['locate ' + '[someone](person)' + ' in the ' + '[' + location + '](destination)' for location in locations_in])

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
tasks_answer_.append(['answer a question' ])
tasks_answer_.append(['answer a question to ' + '[' + name + '](person)' for name in names])
tasks_answer_.append(['answer a question to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_answer_.append(['answer a question to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for name in names for location in locations_in])

tasks_answer_.append(['answer a question to ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_answer_.append(['answer a question to ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_answer_.append(['answer a question to ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations_in])

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
tasks_tell_.append(['tell ' + '[' + w + '](sentence)' + ' to ' + '[' + name + '](person)' for w in what_to_tell_to for name in names])
tasks_tell_.append(['say ' + '[' + w + '](sentence)' + ' to ' + '[' + name + '](person)' for w in what_to_tell_to for name in names])
tasks_tell_.append(['tell ' + '[' + w + '](sentence)' + ' to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for w in what_to_tell_to for name in names for location in locations])
tasks_tell_.append(['say ' + '[' + w + '](sentence)' + ' to ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for w in what_to_tell_to for name in names for location in locations])
tasks_tell_.append(['tell ' + '[' + w + '](sentence)' + ' to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for w in what_to_tell_to for name in names for location in locations_in])
tasks_tell_.append(['say ' + '[' + w + '](sentence)' + ' to ' + '[' + name + '](person)' + ' in the ' + '[' + location + '](destination)' for w in what_to_tell_to for name in names for location in locations_in])

tasks_tell_.append(['tell ' + '[' + w + '](sentence)' + ' to ' + '[' + pronoun + '](person)' for w in what_to_tell_to for pronoun in pronouns])
tasks_tell_.append(['say ' + '[' + w + '](sentence)' + ' to ' + '[' + pronoun + '](person)' for w in what_to_tell_to for pronoun in pronouns])
tasks_tell_.append(['tell ' + '[' + w + '](sentence)' + ' to ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for w in what_to_tell_to for pronoun in pronouns for location in locations])
tasks_tell_.append(['say ' + '[' + w + '](sentence)' + ' to ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for w in what_to_tell_to for pronoun in pronouns for location in locations])
tasks_tell_.append(['tell ' + '[' + w + '](sentence)' + ' to ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for w in what_to_tell_to for pronoun in pronouns for location in locations_in])
tasks_tell_.append(['say ' + '[' + w + '](sentence)' + ' to ' + '[' + pronoun + '](person)' + ' in the ' + '[' + location + '](destination)' for w in what_to_tell_to for pronoun in pronouns for location in locations_in])


tasks_tell_.append(['say ' + '[' + w + '](sentence)' for w in what_to_tell_to])
tasks_tell_.append(['tell ' + '[' + w + '](sentence)' for w in what_to_tell_to])
tasks_tell_.append(['tell [' + pronoun + '](person) ' + '[' + w + '](sentence)' for w in what_to_tell_to for pronoun in pronouns ])

tasks_tell_.append(['tell [' + pronoun + '](person) ' + 'the [name of the person](sentence) at the [' + location + '](destination)' for pronoun in pronouns for location in locations_at])
tasks_tell_.append(['tell [' + pronoun + '](person) ' + 'the [name of the person](sentence) in the [' + location + '](destination)' for pronoun in pronouns for location in locations_in])
tasks_tell_.append(['tell [' + pronoun + '](person) ' + '[how many](sentence) [' + object + '](object) there are on the [' + location + '](destination)' for pronoun in pronouns for object in objects for location in locations_on])

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
tasks_guide_.append(['accompany ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['conduct ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['escort ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['guide ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['lead ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['take ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['oversee ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['supervise ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['usher ' + '[' + pronoun + '](person)'  for pronoun in pronouns])

tasks_guide_.append(['accompany ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['conduct ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['escort ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['guide ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['lead ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['take ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['oversee ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['supervise ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['usher ' + '[' + name + '](person)'  for name in names])

tasks_guide_.append(['accompany ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['conduct ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['escort ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['guide ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['lead ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['take ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['oversee ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['supervise ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['usher ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])

tasks_guide_.append(['accompany ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['conduct ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['escort ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['guide ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['lead ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['take ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['oversee ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['supervise ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['usher ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])

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
tasks_follow_.append(['come after ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_follow_.append(['go after ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_follow_.append(['come behind ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_follow_.append(['go behind ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_follow_.append(['follow ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_follow_.append(['pursue ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_follow_.append(['chase ' + '[' + pronoun + '](person)' for pronoun in pronouns])

tasks_follow_.append(['come after ' + '[' + name + '](person)' for name in names])
tasks_follow_.append(['go after ' + '[' + name + '](person)' for name in names])
tasks_follow_.append(['come behind ' + '[' + name + '](person)' for name in names])
tasks_follow_.append(['go behind ' + '[' + name + '](person)' for name in names])
tasks_follow_.append(['follow ' + '[' + name + '](person)' for name in names])
tasks_follow_.append(['pursue ' + '[' + name + '](person)' for name in names])
tasks_follow_.append(['chase ' + '[' + name + '](person)' for name in names])

tasks_follow_.append(['come after '  + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_follow_.append(['go after ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_follow_.append(['come behind ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_follow_.append(['go behind ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_follow_.append(['follow ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_follow_.append(['pursue ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_follow_.append(['chase ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])

tasks_follow_.append(['come after '  + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_follow_.append(['go after ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_follow_.append(['come behind ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_follow_.append(['go behind ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_follow_.append(['follow ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_follow_.append(['pursue ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_follow_.append(['chase ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)' for name in names for location in locations])

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
tasks_meet_.append(['meet ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_meet_.append(['encounter ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_meet_.append(['face ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_meet_.append(['greet ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_meet_.append(['see ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_meet_.append(['stumble ' + '[' + pronoun + '](person)' for pronoun in pronouns])
tasks_meet_.append(['salute ' + '[' + pronoun + '](person)' for pronoun in pronouns])

tasks_meet_.append(['meet ' + '[' + name + '](person)' for name in names])
tasks_meet_.append(['encounter ' + '[' + name + '](person)' for name in names])
tasks_meet_.append(['face ' + '[' + name + '](person)' for name in names])
tasks_meet_.append(['greet ' + '[' + name + '](person)' for name in names])
tasks_meet_.append(['see ' + '[' + name + '](person)' for name in names])
tasks_meet_.append(['stumble ' + '[' + name + '](person)' for name in names])
tasks_meet_.append(['salute ' + '[' + name + '](person)' for name in names])

tasks_meet_.append(['meet '  + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_meet_.append(['encounter ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_meet_.append(['face ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_meet_.append(['greet ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_meet_.append(['see ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_meet_.append(['stumble ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])
tasks_meet_.append(['salute ' + '[' + pronoun + '](person)' + ' at the ' + '[' + location + '](destination)' for pronoun in pronouns for location in locations])

tasks_meet_.append(['meet '  + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_meet_.append(['encounter ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_meet_.append(['face ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_meet_.append(['greet ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_meet_.append(['see ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_meet_.append(['stumble ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])
tasks_meet_.append(['salute ' + '[' + name + '](person)' + ' at the ' + '[' + location + '](destination)' for name in names for location in locations])

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

print('resampling and appending all the task sentences into one list')

tasks_take_smp = []
tasks_follow_smp = []
tasks_answer_smp = []
tasks_find_smp = []
tasks_guide_smp = []
tasks_tell_smp = []
tasks_go_smp = []
tasks_meet_smp = []

if len(tasks_go)>1:
    # resample
    try: tasks_go = resample(tasks_go, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_go = resample(tasks_go, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_go_smp.append(tasks_go[i])
    # rm temp params
    del tasks_go
if len(tasks_take)>1:
    # resample
    try: tasks_take = resample(tasks_take, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_take = resample(tasks_take, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_take_smp.append(tasks_take[i])
    # rm temp params
    del tasks_take
if len(tasks_find)>1:
    # resample
    try: tasks_find = resample(tasks_find, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_find = resample(tasks_find, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_find_smp.append(tasks_find[i])
    # rm temp params
    del tasks_find
if len(tasks_answer)>1:
    # resample
    try: tasks_answer = resample(tasks_answer, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_answer = resample(tasks_answer, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_answer_smp.append(tasks_answer[i])
    # rm temp params
    del tasks_answer
if len(tasks_tell)>1:
    # resample
    try: tasks_tell = resample(tasks_tell, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_tell = resample(tasks_tell, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_tell_smp.append(tasks_tell[i])
    # rm temp params
    del tasks_tell
if len(tasks_meet)>1:
    # resample
    try: tasks_meet = resample(tasks_meet, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_meet = resample(tasks_meet, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_meet_smp.append(tasks_meet[i])
    # rm temp params
    del tasks_meet
if len(tasks_follow)>1:
    # resample
    try: tasks_follow = resample(tasks_follow, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_follow = resample(tasks_follow, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_follow_smp.append(tasks_follow[i])
    # rm temp params
    del tasks_follow
if len(tasks_guide)>1:
    # resample
    try: tasks_guide = resample(tasks_guide, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_guide = resample(tasks_guide, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_guide_smp.append(tasks_guide[i])
    # rm temp params
    del tasks_guide

print('-----------------------------------------------------')

f = open("gpsr_sentences.md", "w")

f.write('## intent:go\n')

for task in tasks_go_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:take\n')

for task in tasks_take_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:find\n')

for task in tasks_find_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:answer\n')

for task in tasks_answer_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:tell\n')

for task in tasks_tell_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:guide\n')

for task in tasks_guide_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:follow\n')

for task in tasks_follow_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:meet\n')

for task in tasks_meet_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.close()
