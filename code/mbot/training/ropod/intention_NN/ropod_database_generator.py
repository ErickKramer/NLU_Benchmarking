#!/usr/bin/env python3

import random
import yaml
import msgpack
import numpy as np
from sklearn.utils import resample

# load parameters from yaml
# ================================================================================================================
yaml_dict = yaml.load(open('../../../../../ros/config/config_mbot_nlu_training_ropod.yaml'))['intent_train']
random_state = eval(yaml_dict['resample_random_state'])

# params for balancing individual structures
# ================================================================================================================
# number of types of different structured sentences in each of intent classes
n_struct = {'go': 55, 'attach': 16, 'detach': 16, 'push': 3, 'find': 62, 'guide': 42, 'follow': 28}
# number of samples per structe required enough to make balances data
n_samples_per_intent = int(yaml_dict['n_examples']/len(n_struct))
print('Number of intents ', len(n_struct))
print('Number of samples per intent ', n_samples_per_intent)
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
            'tray', 'plate', 'newspaper', 'magazine', 'document', 'station', 'station a', 'box', 'red robot', 'green door']

objects_the = ['cookies', 'almonds', 'book', 'pen', 'notebook', 'laptop', 'tablet', 'charger', 'pencil', 'chips', 'senbei', 'pringles',
                           'peanuts', 'biscuits', 'crackers', 'candies', 'chocolate bar', 'manju', 'mints', 'chewing gums', 'chocolate egg', 'chocolate tablet',
                           'donuts', 'cake', 'pie', 'food', 'peach', 'strawberries', 'grapes', 'blueberries', 'blackberries', 'salt', 'sugar', 'bread', 'cheese',
                           'ham', 'burger', 'lemon', 'onion', 'lemons', 'apples', 'onions', 'orange', 'oranges', 'peaches', 'banana', 'bananas', 'noodles',
                           'apple', 'paprika', 'watermelon', 'sushi', 'pepper', 'pear', 'pizza', 'yogurt', 'drink', 'milk', 'juice', 'coffee', 'hot chocolate',
                           'whisky', 'rum', 'vodka', 'cider', 'lemonade', 'tea', 'water', 'beer', 'coke', 'sprite', 'wine', 'sake', 'toiletries', 'toothpaste',
                           'cream', 'lotion', 'dryer', 'comb', 'towel', 'shampoo','bed', 'soap', 'cloth', 'sponge', 'toilet paper', 'toothbrush', 'container', 'containers',
                           'glass','pasta','can', 'bottle', 'fork', 'knife', 'bowl', 'tray', 'plate', 'newspaper', 'magazine', 'rice','kleenex', 'whiteboard cleaner', 'cup',
                           'document', 'blue robot', 'red door', 'elevator']

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

positions = ['right','left','front','back']

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
pronoun_it = 'it'
people = ['person','people', 'man','woman','boy','girl', 'men','women','guard','nurse','doctor','secretary']
names = list(set(names_male+names_female))

# ================================================================================================================
# intros
# ================================================================================================================
intros = ['robot', 'please', 'could you please', 'robot please', 'robot could you please', 'can you', 'robot can you',  'could you', 'robot could you']


# Prining number of objects used in the generator
print('objects', len(objects))
print('locations', len(sorted(locations, key=str.lower)))
print('names_female', len(names))
print('intros', len(intros))
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

#----------------------------------------attach---------------------------------------------
tasks_attach_.append(['attach to ' + pronoun_it + ' - attach'])
tasks_attach_.append(['connect to ' + pronoun_it + ' - attach'])
tasks_attach_.append(['dock to ' + pronoun_it + ' - attach'])
tasks_attach_.append(['charge in ' + pronoun_it + ' - attach'])

tasks_attach_.append(['attach to ' + pronoun_it + ' from the ' + position + ' - attach' for position in positions])
tasks_attach_.append(['connect to ' + pronoun_it + ' from the ' + position + ' - attach' for position in positions])
tasks_attach_.append(['dock to ' + pronoun_it + ' from the ' + position + ' - attach' for position in positions])
tasks_attach_.append(['charge in ' + pronoun_it + ' from the ' + position + ' - attach' for position in positions])

tasks_attach_.append(['attach to the ' + object + ' from the '+ position + ' - attach' for object in attaching_objects for position in positions])
tasks_attach_.append(['connect to the ' + object +' from the ' + position + ' - attach' for object in attaching_objects for position in positions])
tasks_attach_.append(['dock to the ' + object +' from the ' + position + ' - attach' for object in attaching_objects for position in positions])
tasks_attach_.append(['charge in the ' + object +' from the ' + position + ' - attach' for object in attaching_objects for position in positions])

tasks_attach_.append(['attach to the ' + object + ' ' + indicator +' from the '+ position + ' - attach' for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['connect to the ' + object + ' ' + indicator +' from the '+ position + ' - attach'for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['dock to the ' + object + ' ' + indicator +' from the '+ position + ' - attach'for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['charge in the ' + object + ' ' + indicator +' from the '+ position + ' - attach'for object in attaching_objects for indicator in indicators for position in positions])

tasks_attach_.append(['attach to the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - attach' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['connect to the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - attach'for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['dock to the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - attach'for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['charge in the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - attach'for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])


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
tasks_detach_.append(['detach from ' + pronoun_it + ' - detach'])
tasks_detach_.append(['disconnect from ' + pronoun_it + ' - detach'])
tasks_detach_.append(['undock from ' + pronoun_it + ' - detach'])
tasks_detach_.append(['uncharge from ' + pronoun_it + ' - detach'])

tasks_detach_.append(['detach from ' + pronoun_it + ' from the ' + position + ' - detach' for position in positions])
tasks_detach_.append(['disconnect from ' + pronoun_it + ' from the ' + position + ' - detach' for position in positions])
tasks_detach_.append(['undock from ' + pronoun_it + ' from the ' + position + ' - detach' for position in positions])
tasks_detach_.append(['uncharge from ' + pronoun_it + ' from the ' + position + ' - detach' for position in positions])

tasks_detach_.append(['detach from the ' + object +' from the ' + position  + ' - detach' for object in attaching_objects for position in positions])
tasks_detach_.append(['disconnect from the ' + object +' from the ' + position  + ' - detach' for object in attaching_objects for position in positions])
tasks_detach_.append(['undock from the ' + object +' from the ' + position  + ' - detach' for object in attaching_objects for position in positions])
tasks_detach_.append(['uncharge from the ' + object +' from the ' + position  + ' - detach' for object in attaching_objects for position in positions])

tasks_detach_.append(['detach from the ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['disconnect from the ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['undock from the ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['uncharge from the ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for indicator in indicators for position in positions])

tasks_detach_.append(['detach from the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['disconnect from the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['undock from the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['uncharge from the ' + pre_indicator + ' ' + object + ' ' + indicator +' from the '+ position + ' - detach' for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])

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
tasks_push_.append(['push the ' + object + ' - push' for object in objects])
tasks_push_.append(['push the ' + object + ' from behind - push' for object in objects])
tasks_push_.append(['push the ' + object + ' from the ' + position +' - push' for object in objects for position in positions])
tasks_push_.append(['push the ' + object + ' from the ' + location + ' to the ' + location2 + ' - push' for object in objects for location in locations for location2 in locations])
tasks_push_.append(['push the ' + object + ' from behind to the '+ location +' - push' for object in objects for location in locations])
tasks_push_.append(['push the ' + object + ' from the ' + position +' to the ' + location + ' - push' for object in objects for position in positions for location in locations])

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

tasks_guide_.append(['accompany the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['conduct the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['escort the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['guide the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['lead the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['take the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['oversee the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['supervise the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['usher the ' + person + ' from the '+ location2 +' to the ' + location + ' - guide'  for person in people for location2 in locations for location in locations])

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

tasks_follow_.append(['come after the '  + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['go after the ' + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['come behind the ' + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['go behind the ' + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['follow the ' + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['pursue the ' + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])
tasks_follow_.append(['chase the ' + person + ' from the ' + location2 +' to the ' + location + ' - follow' for person in people for location2 in locations for location in locations])

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

#----------------------------------------------------------------------------------------------
print('resampling and appending all the task sentences into one list')
tasks = []
if len(tasks_go)>1:
    # resample
    try: tasks_go = resample(tasks_go, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_go = resample(tasks_go, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_go[i])
    print('Number of sentences after adding go ', len(tasks))
    # rm temp params
    del tasks_go
if len(tasks_attach)>1:
    # resample
    try: tasks_attach = resample(tasks_attach, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_attach = resample(tasks_attach, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_attach[i])
    print('Number of sentences after adding attach ', len(tasks))
    # rm temp params
    del tasks_attach
if len(tasks_detach)>1:
    # resample
    try: tasks_detach = resample(tasks_detach, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_detach = resample(tasks_detach, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_detach[i])
    print('Number of sentences after adding detach ', len(tasks))
    # rm temp params
    del tasks_detach
if len(tasks_push)>1:
    # resample
    try: tasks_push = resample(tasks_push, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_push = resample(tasks_push, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_push[i])
    print('Number of sentences after adding push ', len(tasks))
    # rm temp params
    del tasks_push
if len(tasks_find)>1:
    # resample
    try: tasks_find = resample(tasks_find, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_find = resample(tasks_find, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_find[i])
    print('Number of sentences after adding find ', len(tasks))
    # rm temp params
    del tasks_find
if len(tasks_follow)>1:
    # resample
    try: tasks_follow = resample(tasks_follow, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_follow = resample(tasks_follow, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_follow[i])
    print('Number of sentences after adding follow ', len(tasks))
    # rm temp params
    del tasks_follow
if len(tasks_guide)>1:
    # resample
    try: tasks_guide = resample(tasks_guide, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_guide = resample(tasks_guide, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
    # append to main list
    for i in range(n_samples_per_intent): tasks.append(tasks_guide[i])
    print('Number of sentences after adding guide ', len(tasks))
    # rm temp params
    del tasks_guide

print('-----------------------------------------------------')

print('I appended {} this many sentences'.format(len(tasks)))

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

# Take out the sentences which are longer than 15 words (The number is choosen by Pedro Martins,
# ref: mbot_nlu/ros/doc/pedro_thesis.pdf)
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
for match in matches: print('---> ', match)
print('-----------------------------------------------------')
print('number of sentences with more than 15 words', matches_lens)
print('Total number of inputs', len(sentences))
print('Total number of outputs', len(outputs))
print('-----------------------------------------------------')
print('Data generation is complete for Intent training, you may start the training by running training_nn_model.py script')
