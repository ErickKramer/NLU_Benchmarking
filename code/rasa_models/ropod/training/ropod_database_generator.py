#!/usr/bin/env python3

import random
import numpy as np
from sklearn.utils import resample

# number of types of different structured sentences in each of intent classes
n_struct = {'go': 66, 'attach': 16, 'detach': 16, 'push': 3, 'find': 62, 'guide': 42, 'follow': 28}

data_slider = 0

random_state = None

n_samples_per_intent = int(200000/len(n_struct))
print('Number of samples per intent', n_samples_per_intent)
# data for creating sentences eg: [names, objects]
# ================================================================================================================
# objects
# ================================================================================================================
objects_a = ['kleenex', 'whiteboard cleaner', 'cup', 'snack', 'cereals bar', 'cookie', 'book', 'pen', 'notebook', 'laptop', 'tablet', 'charger',
            'pencil', 'peanut', 'biscuit', 'candy', 'chocolate bar', 'chewing gum', 'chocolate egg', 'chocolate tablet', 'donuts', 'cake', 'pie',
            'peach', 'strawberry', 'blueberry', 'blackberry', 'burger', 'lemon', 'lemon', 'banana', 'watermelon', 'pepper', 'pear', 'pizza',
            'yogurt', 'drink', 'beer', 'coke', 'sprite', 'sake', 'toothpaste', 'cream', 'lotion', 'dryer', 'comb', 'towel', 'shampoo', 'soap',
            'cloth', 'sponge', 'toothbrush', 'container', 'glass', 'can', 'bottle', 'fork', 'knife', 'bowl',
            'tray', 'plate', 'newspaper', 'magazine', 'document', 'station', 'station a', 'box', 'robot', 'green door']

objects_the = ['cookies', 'almonds', 'book', 'pen', 'notebook', 'laptop', 'tablet', 'charger', 'pencil', 'chips', 'senbei', 'pringles',
                           'peanuts', 'biscuits', 'crackers', 'candies', 'chocolate bar', 'manju', 'mints', 'chewing gums', 'chocolate egg', 'chocolate tablet',
                           'donuts', 'cake', 'pie', 'food', 'peach', 'strawberries', 'grapes', 'blueberries', 'blackberries', 'salt', 'sugar', 'bread', 'cheese',
                           'ham', 'burger', 'lemon', 'onion', 'lemons', 'apples', 'onions', 'orange', 'oranges', 'peaches', 'banana', 'bananas', 'noodles',
                           'apple', 'paprika', 'watermelon', 'sushi', 'pepper', 'pear', 'pizza', 'yogurt', 'drink', 'milk', 'juice', 'coffee', 'hot chocolate',
                           'whisky', 'rum', 'vodka', 'cider', 'lemonade', 'tea', 'water', 'beer', 'coke', 'sprite', 'wine', 'sake', 'toiletries', 'toothpaste',
                           'cream', 'lotion', 'dryer', 'comb', 'towel', 'shampoo','bed', 'soap', 'cloth', 'sponge', 'toilet paper', 'toothbrush', 'container', 'containers',
                           'glass','pasta','can', 'bottle', 'fork', 'knife', 'bowl', 'tray', 'plate', 'newspaper', 'magazine', 'rice','kleenex', 'whiteboard cleaner', 'cup',
                           'document', 'blue robot', 'red door', 'elevator', 'station','desk']
attaching_objects = ['wall', 'station','socket','charger','power outlet','base', 'charging base', 'charging station', 'robot',
                     'trolley']

indicators = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
              '1','2','3','4','5','6','7','8','9','0','blue','red','green','yellow','orange','black','purple','brown','gray','pink']

pre_indicators = ['small','big','free','unused','empty']
objects = list(set(objects_a + objects_the))
# ================================================================================================================
# locations
# ================================================================================================================
locations_on = ['nightstand', 'bookshelf','floor', 'coffee table', 'side table', 'kitchen table', 'kitchen cabinet',
                                'tv stand', 'sofa', 'couch', 'bedroom chair', 'kitchen chair', 'living room table', 'center table',
                                'drawer', 'desk', 'cupboard', 'side shelf', 'bookcase', 'dining table', 'fridge', 'counter',
                                'cabinet', 'table', 'bedchamber', 'chair', 'dryer', 'oven', 'rocking chair', 'stove', 'television', 'bed', 'dressing table',
                                'bench', 'futon', 'beanbag', 'stool', 'sideboard', 'washing machine', 'dishwasher']

locations_in = ['wardrobe', 'nightstand','laboratory', 'bookshelf', 'dining room', 'bedroom', 'closet', 'living room', 'bar', 'office',
                                'drawer', 'kitchen', 'cupboard', 'side shelf', 'refrigerator', 'corridor', 'cabinet', 'bathroom', 'toilet', 'hall', 'hallway',
                                'master bedroom', 'dormitory room', 'bedchamber', 'cellar', 'den', 'garage', 'playroom', 'porch', 'staircase', 'sunroom', 'music room',
                                'prayer room', 'utility room', 'shed', 'basement', 'workshop', 'ballroom', 'box room', 'conservatory', 'drawing room',
                                'games room', 'larder', 'library', 'parlour', 'guestroom', 'crib', 'shower']

locations_at = ['wardrobe', 'nightstand','delivery area', 'bookshelf', 'coffee table', 'side table', 'kitchen table', 'kitchen cabinet',
                                'bed', 'bedside', 'closet', 'tv stand', 'sofa', 'couch', 'bedroom chair', 'kitchen chair',
                                'living room table', 'center table', 'bar', 'drawer', 'desk', 'cupboard', 'sink', 'side shelf',
                                'bookcase', 'dining table', 'refrigerator', 'counter', 'door', 'cabinet', 'table', 'master bedroom', 'dormitory room',
                                'bedchamber', 'chair', 'dryer', 'entrance', 'garden', 'oven', 'rocking chair', 'room', 'stove', 'television', 'washer',
                                'cellar', 'den', 'laundry', 'pantry', 'patio', 'balcony', 'lamp', 'window', 'lawn', 'cloakroom', 'telephone', 'dressing table',
                                'bench', 'futon', 'radiator', 'washing machine', 'dishwasher', 'main entrance','elevator', 'reception']

locations = list(set(locations_at+locations_in+locations_on))

positions = ['right','left','front','back']

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

people = ['person','people', 'man','woman','boy','girl', 'men','women','guard','nurse','doctor','secretary']

pronouns = ['me', 'us', 'him', 'her', 'them']
pronoun_it = 'it'

names = list(set(names_male+names_female))

# Prining number of objects used in the generator
print('objects', len(objects))
print('locations', len(sorted(locations, key=str.lower)))
print('names', len(names))
print('-----------------------------------------------------')

# initiating lists (2 per intent)
tasks_go = []; tasks_go_ = []
tasks_attach = []; tasks_attach_ = []
tasks_detach = []; tasks_detach_ = []
tasks_push = []; tasks_push_ = []
tasks_find = []; tasks_find_ = []
tasks_guide = []; tasks_guide_ = []
tasks_follow = []; tasks_follow_ = []

#------------------------------------------GO----------------------------------------------
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

#----------------------------------------attach---------------------------------------------
tasks_attach_.append(['attach to ' + '[' + pronoun_it + '](object)'])
tasks_attach_.append(['connect to ' + '[' + pronoun_it + '](object)'])
tasks_attach_.append(['dock to ' + '[' + pronoun_it + '](object)'])
tasks_attach_.append(['charge in ' + '[' + pronoun_it + '](object)'])

tasks_attach_.append(['attach to [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])
tasks_attach_.append(['connect to [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])
tasks_attach_.append(['dock to [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])
tasks_attach_.append(['charge in [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])

tasks_attach_.append(['attach to the [' + object + '](object) from the ['+ position + '](position)' for object in attaching_objects for position in positions])
tasks_attach_.append(['connect to the [' + object + '](object) from the [' + position + '](position)' for object in attaching_objects for position in positions])
tasks_attach_.append(['dock to the [' + object + '](object) from the [' + position + '](position)' for object in attaching_objects for position in positions])
tasks_attach_.append(['charge in the [' + object + '](object) from the [' + position + '](position)' for object in attaching_objects for position in positions])

tasks_attach_.append(['attach to the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)'  for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['connect to the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['dock to the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['charge in the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])

tasks_attach_.append(['attach to the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)'  for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['connect to the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['dock to the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['charge in the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])


# resampling and appending individual structures
len_of_str = [len(tasks_attach_[i]) for i in range(len(tasks_attach_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_attach_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_attach_[i] = resample(tasks_attach_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_attach_[i] = resample(tasks_attach_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_attach = [item for sublist in tasks_attach_ for item in sublist]
# rem temp params
del tasks_attach_, len_of_str, mean_of_strct_lens
print ("number of 'attach' sentences", len(tasks_attach))

#----------------------------------------detach---------------------------------------------
tasks_detach_.append(['detach from ' + '[' + pronoun_it + '](object)'])
tasks_detach_.append(['disconnect from ' + '[' + pronoun_it + '](object)'])
tasks_detach_.append(['undock from ' + '[' + pronoun_it + '](object)'])
tasks_detach_.append(['uncharge from ' + '[' + pronoun_it + '](object)'])

tasks_detach_.append(['detach from [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])
tasks_detach_.append(['disconnect from [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])
tasks_detach_.append(['undock from [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])
tasks_detach_.append(['uncharge from [' + pronoun_it + '](object) from the [' + position + '](position)' for position in positions])

tasks_detach_.append(['detach from the [' + object +'](object) from the [' + position  + '](position)' for object in attaching_objects for position in positions])
tasks_detach_.append(['disconnect from the [' + object +'](object) from the [' + position  + '](position)' for object in attaching_objects for position in positions])
tasks_detach_.append(['undock from the [' + object +'](object) from the [' + position  + '](position)' for object in attaching_objects for position in positions])
tasks_detach_.append(['uncharge from the [' + object +'](object) from the [' + position  + '](position)' for object in attaching_objects for position in positions])

tasks_detach_.append(['detach from the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['disconnect from the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['undock from the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['uncharge from the [' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for indicator in indicators for position in positions])

tasks_detach_.append(['detach from the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['disconnect from the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['undock from the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['uncharge from the [' + pre_indicator + ' ' + object + ' ' + indicator +'](object) from the ['+ position + '](position)' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])


# resampling and appending individual structures
len_of_str = [len(tasks_detach_[i]) for i in range(len(tasks_detach_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_detach_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_detach_[i] = resample(tasks_detach_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_detach_[i] = resample(tasks_detach_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_detach = [item for sublist in tasks_detach_ for item in sublist]
# rem temp params
del tasks_detach_, len_of_str, mean_of_strct_lens
print ("number of 'detach' sentences", len(tasks_detach))

#----------------------------------------push---------------------------------------------
tasks_push_.append(['push the ' + '[' + object + '](object)' for object in objects])
tasks_push_.append(['push the ' + '[' + object + '](object) from [behind](position)' for object in objects])
tasks_push_.append(['push the ' + '[' + object + '](object) from the [' + position +'](position)' for object in objects for position in positions])
tasks_push_.append(['push the [' + object + '](object) from the [' + location + '](source) to the [' + location2 + '](destination)' for object in objects for location in locations for location2 in locations])
tasks_push_.append(['push the [' + object + '](object) from [behind](position) to the [' + location + '](destination)' for object in objects for location in locations])
tasks_push_.append(['push the [' + object + '](object) from the [' + position + '](position) to the [' + location + '](destination)' for object in objects for position in positions for location in locations])

# resampling and appending individual structures
len_of_str = [len(tasks_push_[i]) for i in range(len(tasks_push_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_push_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_push_[i] = resample(tasks_push_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_push_[i] = resample(tasks_push_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_push = [item for sublist in tasks_push_ for item in sublist]
# rem temp params
del tasks_push_, len_of_str, mean_of_strct_lens
print ("number of 'push' sentences", len(tasks_push))
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

tasks_find_.append(['find the ' + '[' + person + '](person)' + ' in the ' + '[' + location + '](destination)' for person in people for location in locations_in])
tasks_find_.append(['look for the ' + '[' + person + '](person)' + ' in the ' + '[' + location + '](destination)' for person in people for location in locations_in])
tasks_find_.append(['locate the ' + '[' + person + '](person)' + ' in the ' + '[' + location + '](destination)' for person in people for location in locations_in])
tasks_find_.append(['pinpoint the ' + '[' + person + '](person)' + ' in the ' + '[' + location + '](destination)' for person in people for location in locations_in])
tasks_find_.append(['spot the ' + '[' + person + '](person)' + ' in the ' + '[' + location + '](destination)' for person in people for location in locations_in])

tasks_find_.append(['find the ' + '[' + person + '](person)' + ' at the ' + '[' + location + '](destination)' for person in people for location in locations_at])
tasks_find_.append(['look for the ' + '[' + person + '](person)' + ' at the ' + '[' + location + '](destination)' for person in people for location in locations_at])
tasks_find_.append(['locate the ' + '[' + person + '](person)' + ' at the ' + '[' + location + '](destination)' for person in people for location in locations_at])
tasks_find_.append(['pinpoint the ' + '[' + person + '](person)' + ' at the ' + '[' + location + '](destination)' for person in people for location in locations_at])
tasks_find_.append(['spot the ' + '[' + person + '](person)' + ' at the ' + '[' + location + '](destination)' for person in people for location in locations_at])

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

#---------------------------------------------GUIDE-------------------------------------
tasks_guide_.append(['accompany ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['conduct ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['escort ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['guide ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['lead ' + '[' + pronoun + '](person)'  for pronoun in pronouns])
tasks_guide_.append(['take ' + '[' + pronoun + '](person)'  for pronoun in pronouns])

tasks_guide_.append(['accompany ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['conduct ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['escort ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['guide ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['lead ' + '[' + name + '](person)'  for name in names])
tasks_guide_.append(['take ' + '[' + name + '](person)'  for name in names])

tasks_guide_.append(['accompany ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['conduct ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['escort ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['guide ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['lead ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])
tasks_guide_.append(['take ' + '[' + pronoun + '](person)' + ' to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location in locations])

tasks_guide_.append(['accompany ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['conduct ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['escort ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['guide ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['lead ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])
tasks_guide_.append(['take ' + '[' + name + '](person)' + ' to the ' + '[' + location + '](destination)'  for name in names for location in locations])

tasks_guide_.append(['accompany the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)'  for person in people for location in locations])
tasks_guide_.append(['conduct the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)'  for person in people for location in locations])
tasks_guide_.append(['escort the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)'  for person in people for location in locations])
tasks_guide_.append(['guide the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)'  for person in people for location in locations])
tasks_guide_.append(['lead the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)'  for person in people for location in locations])
tasks_guide_.append(['take the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)'  for person in people for location in locations])

tasks_guide_.append(['accompany ' + '[' + pronoun + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location2 in locations for location in locations])
tasks_guide_.append(['conduct ' + '[' + pronoun + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location2 in locations for location in locations])
tasks_guide_.append(['escort ' + '[' + pronoun + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location2 in locations for location in locations])
tasks_guide_.append(['guide ' + '[' + pronoun + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location2 in locations for location in locations])
tasks_guide_.append(['lead ' + '[' + pronoun + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location2 in locations for location in locations])
tasks_guide_.append(['take ' + '[' + pronoun + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for pronoun in pronouns for location2 in locations for location in locations])

tasks_guide_.append(['accompany ' + '[' + name + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['conduct ' + '[' + name + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['escort ' + '[' + name + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['guide ' + '[' + name + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['lead ' + '[' + name + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['take ' + '[' + name + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for name in names for location2 in locations for location in locations])

tasks_guide_.append(['accompany the ' + '[' + person + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['conduct the ' + '[' + person + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['escort the ' + '[' + person + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['guide the ' + '[' + person + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['lead the ' + '[' + person + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['take the ' + '[' + person + '](person) from the ['+ location2 +'](source) to the ' + '[' + location + '](destination)'  for person in people for location2 in locations for location in locations])

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

tasks_follow_.append(['come after the '  + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])
tasks_follow_.append(['go after the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])
tasks_follow_.append(['come behind the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])
tasks_follow_.append(['go behind the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])
tasks_follow_.append(['follow the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])
tasks_follow_.append(['pursue the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])
tasks_follow_.append(['chase the ' + '[' + person + '](person)' + ' to the ' + '[' + location + '](destination)' for person in people for location in locations])

tasks_follow_.append(['come after '  + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])
tasks_follow_.append(['go after ' + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])
tasks_follow_.append(['come behind ' + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])
tasks_follow_.append(['go behind ' + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])
tasks_follow_.append(['follow ' + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])
tasks_follow_.append(['pursue ' + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])
tasks_follow_.append(['chase ' + '[' + pronoun + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for pronoun in pronouns for location2 in locations for location in locations])

tasks_follow_.append(['come after '  + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])
tasks_follow_.append(['go after ' + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])
tasks_follow_.append(['come behind ' + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])
tasks_follow_.append(['go behind ' + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])
tasks_follow_.append(['follow ' + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])
tasks_follow_.append(['pursue ' + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])
tasks_follow_.append(['chase ' + '[' + name + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for name in names for location2 in locations for location in locations])

tasks_follow_.append(['come after the '  + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['go after the ' + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['come behind the ' + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['go behind the ' + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['follow the ' + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['pursue the ' + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['chase the ' + '[' + person + '](person) from the [' + location2 +'](source) to the ' + '[' + location + '](destination)' for person in people for location2 in locations for location in locations])

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

print('-----------------------------------------------------')

print('resampling and appending all the task sentences into one list')
tasks_go_smp = []
tasks_attach_smp = []
tasks_detach_smp = []
tasks_push_smp = []
tasks_find_smp = []
tasks_guide_smp = []
tasks_follow_smp = []

if len(tasks_go)>1:
    # resample
    try: tasks_go = resample(tasks_go, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_go = resample(tasks_go, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_go_smp.append(tasks_go[i])
    # rm temp params
    del tasks_go
if len(tasks_attach)>1:
    # resample
    try: tasks_attach = resample(tasks_attach, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_attach = resample(tasks_attach, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_attach_smp.append(tasks_attach[i])
    # rm temp params
    del tasks_attach
if len(tasks_detach)>1:
    # resample
    try: tasks_detach = resample(tasks_detach, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_detach = resample(tasks_detach, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_detach_smp.append(tasks_detach[i])
    # rm temp params
    del tasks_detach
if len(tasks_push)>1:
    # resample
    try: tasks_push = resample(tasks_push, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_push = resample(tasks_push, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_push_smp.append(tasks_push[i])
    # rm temp params
    del tasks_push
if len(tasks_find)>1:
    # resample
    try: tasks_find = resample(tasks_find, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_find = resample(tasks_find, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks_find_smp.append(tasks_find[i])
    # rm temp params
    del tasks_find
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
print('Total number of sentences ', len(tasks_go_smp)+len(tasks_attach_smp)+len(tasks_detach_smp)+len(tasks_push_smp)+len(tasks_find_smp)+\
                                    len(tasks_follow_smp)+len(tasks_guide_smp))
print('-----------------------------------------------------')

f = open("ropod_sentences.md", "w")

f.write('## intent:go\n')

for task in tasks_go_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:attach\n')

for task in tasks_attach_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:detach\n')

for task in tasks_detach_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:push\n')

for task in tasks_push_smp:
    f.write('- ' + task + '\n')

f.write('\n')

f.write('## intent:find\n')

for task in tasks_find_smp:
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

f.close()
