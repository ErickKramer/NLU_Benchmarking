#!/usr/bin/env python3

import yaml
import msgpack
import numpy as np
from sklearn.utils import resample

# load parameters from yaml
# ================================================================================================================
yaml_dict = yaml.load(open('../../../../../ros/config/config_mbot_nlu_training.yaml'))['slots_train']
random_state = eval(yaml_dict['resample_random_state'])

# params for balancing individual structures
# ================================================================================================================
# number of types of different structured sentences in each of intent classes
n_struct = {'go': 55, 'attach': 16, 'detach': 16, 'push': 3, 'find': 62, 'guide': 42, 'follow': 28}

# number of samples per structe required enough to make balances data
n_samples_per_intent = int(yaml_dict['n_examples']/len(n_struct))

# data slider. bigger the value, bigger the number of sentences with complex structures(eg: grasp to mia at the kitchen the bottle from the bed room)
# but bigger the repeatation of sentences with smaller structure(eg: go to the kitchen)
data_slider = yaml_dict['data_slider']

# data for creating sentences eg: [names, objects]
# ================================================================================================================
# objects
# ================================================================================================================
objects_a = ['snack -Bobject-', 'cereals -Bobject bar -Iobject-', 'cookie -Bobject-', 'book -Bobject-', 'pen -Bobject-', 'notebook -Bobject-',
            'laptop -Bobject-', 'tablet -Bobject-', 'charger -Bobject-', 'pencil -Bobject-', 'peanut -Bobject-',
            'biscuit -Bobject-', 'candy -Bobject-', 'chocolate -Bobject bar -Iobject-', 'chewing -Bobject- gum -Iobject-',
            'chocolate -Bobject- egg -Iobject-', 'chocolate -Bobject- tablet -Iobject-', 'donuts -Bobject-', 'cake -Bobject-', 'pie -Bobject-',
            'peach -Bobject-', 'strawberry -Bobject-', 'blueberry -Bobject-', 'blackberry -Bobject-', 'burger -Bobject-', 'lemon -Bobject-',
            'banana -Bobject-', 'watermelon -Bobject-', 'pepper -Bobject-', 'pear -Bobject-', 'pizza -Bobject-', 'yogurt -Bobject-',
            'drink -Bobject-', 'beer -Bobject-', 'coke -Bobject-', 'sprite -Bobject-', 'sake -Bobject-', 'toothpaste -Bobject-',
            'cream -Bobject-', 'lotion -Bobject-', 'dryer -Bobject-', 'comb -Bobject-', 'towel -Bobject-', 'shampoo -Bobject-',
            'soap -Bobject-', 'cloth -Bobject-', 'sponge -Bobject-', 'toothbrush -Bobject-', 'container -Bobject-', 'glass -Bobject-',
            'can -Bobject-', 'bottle -Bobject-', 'fork -Bobject-', 'knife -Bobject-', 'bowl -Bobject-', 'tray -Bobject-', 'plate -Bobject-',
            'newspaper -Bobject-', 'station -Bobject- a -Iobject-','station -Bobject-', 'red -Bobject- robot -Iobject-', 'green -Bobject- door -Iobject-',
            'box -Bobject-', 'document -Bobject-','magazine -Bobject-', 'kleenex -Bobject-', 'whiteboard -Bobject- cleaner -Iobject-']

objects_the = ['cookies -Bobject-', 'almonds -Bobject-', 'book -Bobject-', 'pen -Bobject-', 'notebook -Bobject-', 'laptop -Bobject-',
            'tablet -Bobject-', 'charger -Bobject-', 'pencil -Bobject-', 'chips -Bobject-', 'senbei -Bobject-', 'pringles -Bobject-',
            'peanuts -Bobject-', 'biscuits -Bobject-', 'crackers -Bobject-', 'candies -Bobject-', 'chocolate -Bobject bar -Iobject-',
            'manju -Bobject-', 'mints -Bobject-', 'chewing -Bobject- gums -Iobject-', 'chocolate -Bobject- egg -Iobject-',
            'chocolate -Bobject- tablet -Iobject-', 'donuts -Bobject-', 'cake -Bobject-', 'pie -Bobject-', 'food -Bobject-',
            'peach -Bobject-', 'strawberries -Bobject-', 'grapes -Bobject-', 'blueberries -Bobject-', 'blackberries -Bobject-',
            'salt -Bobject-', 'sugar -Bobject-', 'bread -Bobject-', 'cheese -Bobject-', 'ham -Bobject-', 'burger -Bobject-', 'ham -Bobject- burger -Iobject-',
            'lemon -Bobject-', 'onion -Bobject-', 'lemons -Bobject-', 'apples -Bobject-', 'onions -Bobject-', 'orange -Bobject-', 'oranges -Bobject-',
            'peaches -Bobject-', 'banana -Bobject-', 'bananas -Bobject-', 'noodles -Bobject-', 'apple -Bobject-', 'paprika -Bobject-',
            'watermelon -Bobject-', 'sushi -Bobject-', 'pepper -Bobject-', 'pear -Bobject-', 'pizza -Bobject-', 'yogurt -Bobject-',
            'drink -Bobject-', 'milk -Bobject-', 'juice -Bobject-', 'coffee -Bobject-', 'hot -Bobject- chocolate', 'whisky -Bobject-',
            'rum -Bobject-', 'vodka -Bobject-', 'cider -Bobject-', 'lemonade -Bobject-', 'tea -Bobject-', 'water -Bobject-', 'beer -Bobject-',
            'coke -Bobject-', 'sprite -Bobject-', 'wine -Bobject-', 'sake -Bobject-', 'toiletries -Bobject-', 'toothpaste -Bobject-',
            'cream -Bobject-', 'lotion -Bobject-', 'dryer -Bobject-', 'comb -Bobject-', 'towel -Bobject-', 'shampoo -Bobject-', 'soap -Bobject-',
            'cloth -Bobject-', 'sponge -Bobject-', 'toilet -Bobject- paper -Iobject-', 'toothbrush -Bobject-', 'container -Bobject-', 'containers -Bobject-',
            'glass -Bobject-', 'can -Bobject-', 'bottle -Bobject-', 'fork -Bobject-', 'knife -Bobject-', 'bowl -Bobject-', 'tray -Bobject-',
            'plate -Bobject-', 'newspaper -Bobject-', 'magazine -Bobject-', 'rice -Bobject-','kleenex -Bobject-',
            'whiteboard -Bobject- cleaner -Iobject-', 'cup -Bobject-','pasta -Bobject-','document', 'blue -Bobject- robot -Iobject-', 'red -Bobject- door -Iobject-',
            'elevator -Bobject-', 'station -Bobject-']

attaching_objects = ['wall -Bobject-', 'station -Bobject-','socket -Bobject-','charger -Bobject-','power -Bobject- outlet -Iobject-',
                    'base -Bobject-', 'charging -Bobject- base -Iobject-', 'charging -Bobject- station -Iobject-', 'robot -Bobject-',
                     'trolley -Bobject-']

indicators = ['a -Iobject-','b -Iobject-','c -Iobject-','d -Iobject-','e -Iobject-','f -Iobject-','g -Iobject-','h -Iobject-','i -Iobject-',
              'j -Iobject-','k -Iobject-','l -Iobject-','m -Iobject-','n -Iobject-','o -Iobject-','p -Iobject-','q -Iobject-','r -Iobject-',
              's -Iobject-','t -Iobject-','u -Iobject-','v -Iobject-','w -Iobject-','x -Iobject-','y -Iobject-','z -Iobject-',
              '1 -Iobject-','2 -Iobject-','3 -Iobject-','4 -Iobject-','5 -Iobject-','6 -Iobject-','7 -Iobject-','8 -Iobject-','9 -Iobject-',
              '0 -Iobject-','blue -Iobject-','red -Iobject-','green -Iobject-','yellow -Iobject-','orange -Iobject-','black -Iobject-','purple -Iobject-',
              'brown -Iobject-','gray -Iobject-','pink -Iobject-']
pre_indicators = ['small -Bobject-','big -Bobject-','free -Bobject-','unused -Bobject-','empty -Bobject-']

objects = list(set(objects_a + objects_the))
# ================================================================================================================
# locations
# ================================================================================================================
locations_on = ['nightstand -Blocation-', 'bookshelf -Blocation-', 'coffee -Blocation- table -Ilocation-', 'side -Blocation- table -Ilocation-',
                'kitchen -Blocation- table -Ilocation-', 'kitchen -Blocation- cabinet -Ilocation-', 'tv -Blocation- stand -Ilocation-',
                'sofa -Blocation-', 'couch -Blocation-', 'bedroom -Blocation- chair -Ilocation-', 'kitchen -Blocation- chair -Ilocation-',
                'living -Blocation- room -Ilocation- table -Ilocation-', 'center -Blocation- table -Ilocation-', 'drawer -Blocation-', 'desk -Blocation-',
                'cupboard -Blocation-', 'side -Blocation- shelf -Ilocation-', 'bookcase -Blocation-', 'dining -Blocation- table -Ilocation-',
                'fridge -Blocation-', 'counter -Blocation-', 'cabinet -Blocation-', 'table -Blocation-', 'bedchamber -Blocation-', 'chair -Blocation-',
                'dryer -Blocation-', 'oven -Blocation-', 'rocking -Blocation- chair -Ilocation-', 'stove -Blocation-', 'television -Blocation-',
                'dressing -Blocation- table -Ilocation-', 'bench -Blocation-', 'futon -Blocation-', 'beanbag -Blocation-', 'stool -Blocation-',
                'sideboard -Blocation-', 'washing -Blocation- machine -Ilocation-', 'dishwasher -Blocation-']

locations_in = ['wardrobe -Blocation-', 'nightstand -Blocation-', 'bookshelf -Blocation-', 'dining -Blocation- room -Ilocation-', 'bedroom -Blocation-',
                'closet -Blocation-', 'living -Blocation- room -Ilocation-', 'bar -Blocation-', 'office -Blocation-', 'drawer -Blocation-',
                'kitchen -Blocation-', 'cupboard -Blocation-', 'side -Blocation- shelf -Ilocation-', 'fridge -Blocation-', 'corridor -Blocation-',
                'cabinet -Blocation-', 'bathroom -Blocation-', 'toilet -Blocation-', 'hall -Blocation-', 'hallway -Blocation-',
                'master -Blocation- bedroom -Ilocation-', 'dormitory -Blocation- room -Ilocation-', 'bedchamber -Blocation-', 'cellar -Blocation-',
                'den -Blocation-', 'garage -Blocation-', 'playroom -Blocation-', 'porch -Blocation-', 'staircase -Blocation-',
                'sun -Blocation- room -Ilocation-', 'music -Blocation- room -Ilocation-', 'prayer -Blocation- room -Ilocation-',
                'utility -Blocation- room -Ilocation-', 'shed -Blocation-', 'basement -Blocation-', 'workshop -Blocation-',
                'ballroom -Blocation-', 'box -Blocation- room -Ilocation-', 'conservatory -Blocation-', 'drawing -Blocation- room -Ilocation-',
                'games -Blocation- room -Ilocation-', 'larder -Blocation-', 'library -Blocation-', 'parlour -Blocation-', 'guestroom -Blocation-',
                'crib -Blocation-', 'shower -Blocation-']

locations_at = ['wardrobe -Blocation-', 'nightstand -Blocation-', 'bookshelf -Blocation-', 'coffee -Blocation- table -Ilocation-',
                'side -Blocation- table -Ilocation-', 'kitchen -Blocation- table -Ilocation-', 'kitchen -Blocation- cabinet -Ilocation-',
                'bed -Blocation-', 'bedside -Blocation-', 'closet -Blocation-', 'tv -Blocation- stand -Ilocation-', 'sofa -Blocation-',
                'couch -Blocation-', 'bedroom -Blocation- chair -Ilocation-', 'kitchen -Blocation- chair -Ilocation-',
                'living -Blocation- room -Ilocation- table -Ilocation-', 'center -Blocation- table -Ilocation-', 'bar -Blocation-',
                'drawer -Blocation-', 'desk -Blocation-', 'cupboard -Blocation-', 'sink -Blocation-', 'side -Blocation- shelf -Ilocation-',
                'bookcase -Blocation-', 'dining -Blocation- table -Ilocation-', 'fridge -Blocation-', 'counter -Blocation-', 'door -Blocation-',
                'cabinet -Blocation-', 'table -Blocation-', 'master -Blocation- bedroom -Ilocation-', 'dormitory -Blocation- room -Ilocation-',
                'bedchamber -Blocation-', 'chair -Blocation-', 'dryer -Blocation-', 'entrance -Blocation-', 'garden -Blocation-',
                'oven -Blocation-', 'rocking -Blocation- chair -Ilocation-', 'room -Blocation-', 'stove -Blocation-', 'television -Blocation-',
                'washer -Blocation-', 'cellar -Blocation-', 'den -Blocation-', 'laundry -Blocation-', 'pantry -Blocation-', 'patio -Blocation-',
                'balcony -Blocation-', 'lamp -Blocation-', 'window -Blocation-', 'lawn -Blocation-', 'cloakroom -Blocation-', 'telephone -Blocation-',
                'dressing -Blocation- table -Ilocation-', 'bench -Blocation-', 'futon -Blocation-', 'radiator -Blocation-',
                'washing -Blocation- machine -Ilocation-', 'dishwasher -Blocation-','main -Blocation- entrance -Ilocation-','elevator -Blocation-', 'reception -Blocation-']

locations = list(set(locations_at+locations_in+locations_on))
positions = ['right -Bposition-','left -Bposition-','front -Bposition-','back -Bposition-']
# ================================================================================================================
# names
# ================================================================================================================
names_female = ['hanna -Bperson-', 'barbara -Bperson-', 'samantha -Bperson-', 'erika -Bperson-', 'sophie -Bperson-', 'jackie -Bperson-',
                'skyler -Bperson-', 'jane -Bperson-', 'olivia -Bperson-', 'emily -Bperson-', 'amelia -Bperson-', 'lily -Bperson-',
                'grace -Bperson-', 'ella -Bperson-', 'scarlett -Bperson-', 'isabelle -Bperson-', 'charlotte -Bperson-', 'daisy -Bperson-',
                'sienna -Bperson-', 'chloe -Bperson-', 'alice -Bperson-', 'lucy -Bperson-', 'florence -Bperson-', 'rosie -Bperson-',
                'amelie -Bperson-', 'eleanor -Bperson-', 'emilia -Bperson-', 'amber -Bperson-', 'ivy -Bperson-', 'brooke -Bperson-',
                'summer -Bperson-', 'emma -Bperson-', 'rose -Bperson-', 'martha -Bperson-', 'faith -Bperson-', 'amy -Bperson-',
                'mia -Bperson-', 'sophia -Bperson-', 'abigail -Bperson-', 'isabella -Bperson-', 'ava -Bperson-',
                'katie -Bperson-', 'argentina -Bperson-', 'madison -Bperson-', 'sarah -Bperson-', 'zoe -Bperson-', 'paige -Bperson-']

names_male = ['ken -Bperson-', 'erick -Bperson-', 'samuel -Bperson-', 'skyler -Bperson-', 'brian -Bperson-', 'thomas -Bperson-',
            'edward -Bperson-', 'michael -Bperson-', 'charlie -Bperson-', 'alex -Bperson-', 'john -Bperson-', 'james -Bperson-',
            'oscar -Bperson-', 'peter -Bperson-', 'oliver -Bperson-', 'jack -Bperson-', 'harry -Bperson-', 'henry -Bperson-',
            'jacob -Bperson-', 'thomas -Bperson-', 'william -Bperson-', 'will -Bperson-', 'joshua -Bperson-', 'josh -Bperson-',
            'noah -Bperson-', 'ethan -Bperson-', 'joseph -Bperson-', 'samuel -Bperson-', 'daniel -Bperson-', 'max -Bperson-',
            'logan -Bperson-', 'isaac -Bperson-', 'dylan -Bperson-', 'freddie -Bperson-', 'tyler -Bperson-', 'harrison -Bperson-',
            'adam -Bperson-', 'theo -Bperson-', 'arthur -Bperson-', 'toby -Bperson-', 'luke -Bperson-', 'lewis -Bperson-',
            'matthew -Bperson-', 'harvey -Bperson-', 'ryan -Bperson-', 'tommy -Bperson-', 'michael -Bperson-', 'nathan -Bperson-',
            'blake -Bperson-', 'charles -Bperson-', 'connor -Bperson-', 'jamie -Bperson-', 'elliot -Bperson-', 'louis -Bperson-',
            'liam -Bperson-', 'mason -Bperson-', 'alexander -Bperson-', 'madison -Bperson-',
            'aaron -Bperson-', 'evan -Bperson-', 'seth -Bperson-','patrick -Bperson-']

people = ['person -Bperson-', 'people -Bperson-', 'man -Bperson-','woman -Bperson-','boy -Bperson-','girl -Bperson-', 'men -Bperson-','women -Bperson-']

pronouns = ['me -Bperson-', 'us -Bperson-', 'him -Bperson-', 'her -Bperson-', 'them -Bperson-']
pronoun_it = 'it -Bobject-'

names = list(set(names_male+names_female))

# ================================================================================================================
# introductions
# ================================================================================================================
intros = ['robot', 'hello robot', 'hello', 'please', 'could you please', 'robot please', 'can you', 'robot can you', 'robot could you', 'could you']

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
tasks_go_.append(['go to ' + name for name in names])
tasks_go_.append(['navigate to ' + name for name in names])
tasks_go_.append(['proceed to ' + name for name in names])
tasks_go_.append(['move to ' + name for name in names])
tasks_go_.append(['advance to ' + name for name in names])
tasks_go_.append(['travel to ' + name for name in names])
tasks_go_.append(['drive to ' + name for name in names])
tasks_go_.append(['come to ' + name for name in names])
tasks_go_.append(['go near ' + name for name in names])
tasks_go_.append(['walk to ' + name for name in names])
tasks_go_.append(['reach ' + name for name in names])

tasks_go_.append(['go to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['navigate to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['proceed to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['move to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['advance to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['travel to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['drive to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['come to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['go near ' + pronoun for pronoun in pronouns])
tasks_go_.append(['walk to ' + pronoun for pronoun in pronouns])
tasks_go_.append(['reach ' + pronoun for pronoun in pronouns])

tasks_go_.append(['go to ' + pronoun_it])
tasks_go_.append(['navigate to ' + pronoun_it])
tasks_go_.append(['proceed to ' + pronoun_it])
tasks_go_.append(['move to ' + pronoun_it])
tasks_go_.append(['advance to ' + pronoun_it])
tasks_go_.append(['travel to ' + pronoun_it])
tasks_go_.append(['drive to ' + pronoun_it])
tasks_go_.append(['come to ' + pronoun_it])
tasks_go_.append(['go near ' + pronoun_it])
tasks_go_.append(['walk to ' + pronoun_it])
tasks_go_.append(['reach ' + pronoun_it])

tasks_go_.append(['go to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['navigate to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['proceed to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['move to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['advance to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['travel to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['drive to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['come to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['go near the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['walk to the ' + location.replace('location', 'destination') for location in locations])
tasks_go_.append(['reach the ' + location.replace('location', 'destination') for location in locations])

tasks_go_.append(['go to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['navigate to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['proceed to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['move to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['advance to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['travel to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['drive to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['come to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['go near ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['walk to ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_go_.append(['reach ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations])

tasks_go_.append(['go to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['navigate to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['proceed to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['move to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['advance to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['travel to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['drive to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['come to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['go near ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['walk to ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_go_.append(['reach ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])

# resampling and appending individual structures
len_of_str = [len(tasks_go_[i]) for i in range(len(tasks_go_))]
mean_of_strct_lens = int(np.mean(len_of_str)) + data_slider
for i in range(len(tasks_go_)):
    # resample if len not enough or more
    if len_of_str[i]!=mean_of_strct_lens:
        try: tasks_go_[i] = resample(tasks_go_[i], n_samples=mean_of_strct_lens, replace=False)
        except: tasks_go_[i] = resample(tasks_go_[i], n_samples=mean_of_strct_lens, replace=True)
# flat list
tasks_go = [item for sublist in tasks_go_ for item in sublist]
# rem temp params
del tasks_go_, len_of_str, mean_of_strct_lens
print ("number of 'go' sentences", len(tasks_go))

#----------------------------------------attach---------------------------------------------
tasks_attach_.append(['attach to ' + pronoun_it])
tasks_attach_.append(['connect to ' + pronoun_it])
tasks_attach_.append(['dock to ' + pronoun_it])
tasks_attach_.append(['charge in ' + pronoun_it])

tasks_attach_.append(['attach to ' + pronoun_it + ' from the ' + position  for position in positions])
tasks_attach_.append(['connect to ' + pronoun_it + ' from the ' + position  for position in positions])
tasks_attach_.append(['dock to ' + pronoun_it + ' from the ' + position  for position in positions])
tasks_attach_.append(['charge in ' + pronoun_it + ' from the ' + position  for position in positions])

tasks_attach_.append(['attach to the ' + object + ' from the '+ position  for object in attaching_objects for position in positions])
tasks_attach_.append(['connect to the ' + object +' from the ' + position  for object in attaching_objects for position in positions])
tasks_attach_.append(['dock to the ' + object +' from the ' + position  for object in attaching_objects for position in positions])
tasks_attach_.append(['charge in the ' + object +' from the ' + position  for object in attaching_objects for position in positions])

tasks_attach_.append(['attach to the ' + object + ' ' + indicator +' from the '+ position  for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['connect to the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['dock to the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])
tasks_attach_.append(['charge in the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])

tasks_attach_.append(['attach to the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['connect to the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['dock to the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_attach_.append(['charge in the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])

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
tasks_detach_.append(['detach from ' + pronoun_it])
tasks_detach_.append(['disconnect from ' + pronoun_it])
tasks_detach_.append(['undock from ' + pronoun_it])
tasks_detach_.append(['uncharge from ' + pronoun_it])

tasks_detach_.append(['detach from ' + pronoun_it + ' from the ' + position for position in positions])
tasks_detach_.append(['disconnect from ' + pronoun_it + ' from the ' + position for position in positions])
tasks_detach_.append(['undock from ' + pronoun_it + ' from the ' + position for position in positions])
tasks_detach_.append(['uncharge from ' + pronoun_it + ' from the ' + position for position in positions])

tasks_detach_.append(['detach from the ' + object +' from the ' + position  for object in attaching_objects for position in positions])
tasks_detach_.append(['disconnect from the ' + object +' from the ' + position  for object in attaching_objects for position in positions])
tasks_detach_.append(['undock from the ' + object +' from the ' + position  for object in attaching_objects for position in positions])
tasks_detach_.append(['uncharge from the ' + object +' from the ' + position  for object in attaching_objects for position in positions])

tasks_detach_.append(['detach from the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['disconnect from the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['undock from the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])
tasks_detach_.append(['uncharge from the ' + object + ' ' + indicator +' from the '+ position for object in attaching_objects for indicator in indicators for position in positions])

tasks_detach_.append(['detach from the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['disconnect from the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['undock from the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])
tasks_detach_.append(['uncharge from the ' + pre_indicator + ' ' + object.replace('Bobject', 'Iobject') + ' ' + indicator +' from the '+ position for object in attaching_objects for pre_indicator in pre_indicators for indicator in indicators for position in positions])


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
tasks_push_.append(['push the '+ object for object in objects])
tasks_push_.append(['push the '+ object + ' from behind -Bposition-' for object in objects])
tasks_push_.append(['push the '+ object + ' from the ' + position for object in objects for position in positions])
tasks_push_.append(['push the ' + object + ' from the ' + location.replace('location', 'source') + ' to the ' + location2.replace('location', 'destination') for object in objects for location in locations for location2 in locations])
tasks_push_.append(['push the ' + object + ' from behind -Bposition- to the '+ location.replace('location', 'destination') for object in objects for location in locations])
tasks_push_.append(['push the ' + object + ' from the ' + position +' to the ' + location.replace('location', 'destination') for object in objects for position in positions for location in locations])

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

tasks_find_.append(['find the ' + object for object in objects])
tasks_find_.append(['look for the ' + object for object in objects])
tasks_find_.append(['locate the ' + object for object in objects])
tasks_find_.append(['pinpoint the ' + object for object in objects])
tasks_find_.append(['spot the ' + object for object in objects])

tasks_find_.append(['find ' + pronoun_it])
tasks_find_.append(['look for ' + pronoun_it])
tasks_find_.append(['locate ' + pronoun_it])
tasks_find_.append(['pinpoint ' + pronoun_it])
tasks_find_.append(['spot ' + pronoun_it])

tasks_find_.append(['find the ' + object + ' in the ' + location.replace('location', 'destination') for object in objects for location in locations_in])
tasks_find_.append(['look for the ' + object + ' in the ' + location.replace('location', 'destination') for object in objects for location in locations_in])
tasks_find_.append(['locate the ' + object + ' in the ' + location.replace('location', 'destination') for object in objects for location in locations_in])
tasks_find_.append(['pinpoint the ' + object + ' in the ' + location.replace('location', 'destination') for object in objects for location in locations_in])
tasks_find_.append(['spot the ' + object + ' in the ' + location.replace('location', 'destination') for object in objects for location in locations_in])

tasks_find_.append(['find ' + pronoun_it + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['look for ' + pronoun_it + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['locate ' + pronoun_it + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['pinpoint ' + pronoun_it + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['spot ' + pronoun_it + ' in the ' + location.replace('location', 'destination') for location in locations_in])

tasks_find_.append(['find ' + name for name in names])
tasks_find_.append(['look for ' + name for name in names])
tasks_find_.append(['locate ' + name for name in names])
tasks_find_.append(['pinpoint ' + name for name in names])
tasks_find_.append(['spot ' + name for name in names])

tasks_find_.append(['find ' + pronoun for pronoun in pronouns])
tasks_find_.append(['look for ' + pronoun for pronoun in pronouns])
tasks_find_.append(['locate ' + pronoun for pronoun in pronouns])
tasks_find_.append(['pinpoint ' + pronoun for pronoun in pronouns])
tasks_find_.append(['spot ' + pronoun for pronoun in pronouns])

tasks_find_.append(['find ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['look for ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['locate ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['pinpoint ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['spot ' + name + ' in the ' + location.replace('location', 'destination') for name in names for location in locations_in])

tasks_find_.append(['find ' + pronoun + ' in the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['look for ' + pronoun + ' in the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['locate ' + pronoun + ' in the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['pinpoint ' + pronoun + ' in the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['spot ' + pronoun + ' in the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])

tasks_find_.append(['find ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['look for ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['locate ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['pinpoint ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations_in])
tasks_find_.append(['spot ' + name + ' at the ' + location.replace('location', 'destination') for name in names for location in locations_in])

tasks_find_.append(['find ' + pronoun + ' at the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['look for ' + pronoun + ' at the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['locate ' + pronoun + ' at the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['pinpoint ' + pronoun + ' at the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])
tasks_find_.append(['spot ' + pronoun + ' at the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations_in])

tasks_find_.append(['find someone -Bperson-'])
tasks_find_.append(['locate someone -Bperson-'])
tasks_find_.append(['look for someone -Bperson-'])
tasks_find_.append(['find a person -Bperson-'])
tasks_find_.append(['locate a person -Bperson-'])
tasks_find_.append(['look for a person -Bperson-'])

tasks_find_.append(['find a person -Bperson-' + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['locate a person -Bperson-' + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['look for a person -Bperson-' + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['find someone -Bperson-' + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['look for someone -Bperson-' + ' in the ' + location.replace('location', 'destination') for location in locations_in])
tasks_find_.append(['locate someone -Bperson-' + ' in the ' + location.replace('location', 'destination') for location in locations_in])

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
tasks_guide_.append(['accompany ' + pronoun for pronoun in pronouns])
tasks_guide_.append(['conduct ' + pronoun for pronoun in pronouns])
tasks_guide_.append(['escort ' + pronoun for pronoun in pronouns])
tasks_guide_.append(['guide ' + pronoun for pronoun in pronouns])
tasks_guide_.append(['lead ' + pronoun for pronoun in pronouns])
tasks_guide_.append(['take ' + pronoun for pronoun in pronouns])

tasks_guide_.append(['accompany ' + name for name in names])
tasks_guide_.append(['conduct ' + name for name in names])
tasks_guide_.append(['escort ' + name for name in names])
tasks_guide_.append(['guide ' + name for name in names])
tasks_guide_.append(['lead ' + name for name in names])
tasks_guide_.append(['take ' + name for name in names])

tasks_guide_.append(['accompany ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_guide_.append(['conduct ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_guide_.append(['escort ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_guide_.append(['guide ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_guide_.append(['lead ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_guide_.append(['take ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])

tasks_guide_.append(['accompany ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_guide_.append(['conduct ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_guide_.append(['escort ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_guide_.append(['guide ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_guide_.append(['lead ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_guide_.append(['take ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])

tasks_guide_.append(['accompany ' + name + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['conduct ' + name + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['escort ' + name + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['guide ' + name + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['lead ' + name + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for name in names for location2 in locations for location in locations])
tasks_guide_.append(['take ' + name + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for name in names for location2 in locations for location in locations])

tasks_guide_.append(['accompany the ' + person + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['conduct the ' + person + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['escort the ' + person + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['guide the ' + person + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['lead the ' + person + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for person in people for location2 in locations for location in locations])
tasks_guide_.append(['take the ' + person + ' from the '+ location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination')  for person in people for location2 in locations for location in locations])

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
tasks_follow_.append(['come after ' + pronoun for pronoun in pronouns])
tasks_follow_.append(['go after ' + pronoun for pronoun in pronouns])
tasks_follow_.append(['come behind ' + pronoun for pronoun in pronouns])
tasks_follow_.append(['go behind ' + pronoun for pronoun in pronouns])
tasks_follow_.append(['follow ' + pronoun for pronoun in pronouns])
tasks_follow_.append(['pursue ' + pronoun for pronoun in pronouns])
tasks_follow_.append(['chase ' + pronoun for pronoun in pronouns])

tasks_follow_.append(['come after ' + name for name in names])
tasks_follow_.append(['go after ' + name for name in names])
tasks_follow_.append(['come behind ' + name for name in names])
tasks_follow_.append(['go behind ' + name for name in names])
tasks_follow_.append(['follow ' + name for name in names])
tasks_follow_.append(['pursue ' + name for name in names])
tasks_follow_.append(['chase ' + name for name in names])

tasks_follow_.append(['come after '  + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_follow_.append(['go after ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_follow_.append(['come behind ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_follow_.append(['go behind ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_follow_.append(['follow ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_follow_.append(['pursue ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])
tasks_follow_.append(['chase ' + pronoun + ' to the ' + location.replace('location', 'destination') for pronoun in pronouns for location in locations])

tasks_follow_.append(['come after '  + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_follow_.append(['go after ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_follow_.append(['come behind ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_follow_.append(['go behind ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_follow_.append(['follow ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_follow_.append(['pursue ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])
tasks_follow_.append(['chase ' + name + ' to the ' + location.replace('location', 'destination') for name in names for location in locations])

tasks_follow_.append(['come after '  + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])
tasks_follow_.append(['go after ' + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])
tasks_follow_.append(['come behind ' + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])
tasks_follow_.append(['go behind ' + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])
tasks_follow_.append(['follow ' + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])
tasks_follow_.append(['pursue ' + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])
tasks_follow_.append(['chase ' + name + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for name in names for location2 in locations for location in locations])

tasks_follow_.append(['come after the '  + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])
tasks_follow_.append(['go after the ' + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])
tasks_follow_.append(['come behind the ' + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])
tasks_follow_.append(['go behind the ' + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])
tasks_follow_.append(['follow the ' + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])
tasks_follow_.append(['pursue the ' + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])
tasks_follow_.append(['chase the ' + person + ' from the ' + location2.replace('location', 'source') +' to the ' + location.replace('location', 'destination') for person in people for location2 in locations for location in locations])

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
print('resampling major classes')
if len(tasks_go)>1:
    try: tasks_go = resample(tasks_go, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_go = resample(tasks_go, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
if len(tasks_attach)>1:
    try: tasks_attach = resample(tasks_attach, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_attach = resample(tasks_attach, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
if len(tasks_detach)>1:
    try: tasks_detach = resample(tasks_detach, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_detach = resample(tasks_detach, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
if len(tasks_push)>1:
    try: tasks_push = resample(tasks_push, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_push = resample(tasks_push, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
if len(tasks_find)>1:
    try: tasks_find = resample(tasks_find, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_find = resample(tasks_find, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
if len(tasks_follow)>1:
    try: tasks_follow = resample(tasks_follow, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_follow = resample(tasks_follow, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
if len(tasks_guide)>1:
    try: tasks_guide = resample(tasks_guide, replace=False, n_samples=n_samples_per_intent, random_state=random_state)
    except ValueError: tasks_guide = resample(tasks_guide, replace=True, n_samples=n_samples_per_intent, random_state=random_state)
print('-----------------------------------------------------')
print('appending all the task sentences into one list')

tasks = []
for i in range(n_samples_per_intent):

    if len(tasks_go)>1: tasks.append(tasks_go[i])

    if len(tasks_attach)>1: tasks.append(tasks_attach[i])

    if len(tasks_detach)>1: tasks.append(tasks_detach[i])

    if len(tasks_push)>1: tasks.append(tasks_push[i])

    if len(tasks_find)>1: tasks.append(tasks_find[i])

    if len(tasks_follow)>1: tasks.append(tasks_follow[i])

    if len(tasks_guide)>1: tasks.append(tasks_guide[i])

print('-----------------------------------------------------')

# Appending introductions (eg: "hello robot" , "could you please" etc.) and generating inputs and outputs
c = 0
sentences = []
outputs = []
for v in range(len(tasks)):

    try:
        task = tasks[v].split(' ')
    except:
        print('error at {}, check if the items in tasks are strings tasks[v] = {}'.format(v, tasks[v]))
        break

    intro = intros[c].split(' ')

    sentence = []
    output = []
    # appending intro if the total length is less than or equal to 15 (ref: Pedro master thesis)
    task_with_intro = []
    if v%4 == 0 and (len(intro) + len(task)) <= 15:
        for x in intro:
            task_with_intro.append(x)
        c = c + 1

    # reinitiating intros
    if c == len(intros):
       c = 0

    # appending task
    for x in task:
        task_with_intro.append(x)

    # appending the task with introduction to the sentence list
    for h in range(len(task_with_intro)):
        if not task_with_intro[h].startswith('-'):
            sentence.append(task_with_intro[h])
            # appending outputs according to the input
            if h < len(task_with_intro)-1:
                if task_with_intro[h+1].startswith('-'):
                    l = task_with_intro[h+1]
                    l = l.replace('-', '')
                    output.append(l)
                else:
                    output.append('O')
            else:
                output.append('O')
    # split sentences and tags
    sentences.append(sentence)
    outputs.append(output)

print('I found this many sentences {}'.format(len(sentences)))
print('I found this many outputs {}'.format(len(outputs)))
# pickling inputs and labels
with open('inputs_slot_filling', 'wb') as inputs_file:
    msgpack.dump(sentences, inputs_file)

with open('outputs_slot_filling', 'wb') as outputs_file:
    msgpack.dump(outputs, outputs_file)

print('Total number of inputs', len(sentences))
print('Total number of outputs', len(outputs))

print('-----------------------------------------------------')

print('Data generation is complete for Slots training, you may start the training by running training_nn_model.py script')
