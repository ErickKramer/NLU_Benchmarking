# Benchmarking of Natural Language Understanding models in context of service robots
This repository contains the code, datasets, and suplemental materials for "Tell Your Robot What To Do: Evaluation of Natural Language Models for Robot Command Processing" by Erick Romero Kramer, Argentina Ortega Sainz Alex Mitrevski Paul G. Ploeger.

## Models used for Benchmarking
* Mbot - [Repository](https://github.com/socrob/mbot_natural_language_processing)
* Rasa - [Documentation](http://rasa.com/docs/rasa/) - [Repository](https://github.com/RasaHQ/rasa)
* LU4R - [Project Site](http://sag.art.uniroma2.it/lu4r.html)
* ECG  - [Repository](https://github.com/icsi-berkeley/ecg_framework_code)

## Repository structure
* code/
    - ecg_framework_code
        - Required repository for ECG
    - ecg_grammars
        - Repository containing the grammars generated for our Benchmarking
    - ecg_robot_code
        - Modified version of the original implementation to extract the semantic specifications
    - ecg_workbench_release
        - Repository containing the tool used to generated ECG grammar files 
    - lu4r
        - Empty folder
    - rasa_models
        - Folder containing the nlu models and scripts to generate the training datasets

* datasets/
    - GPSR category 1 (Single and multiple actions)
    - GPSR category 2 (Single and multiple actions)
    - [ROPOD](https://cordis.europa.eu/project/rcn/206247/en)
* paper.pdf - **PENDING**
* R&D Report
    - Detailed information about the related work presented in the paper.
    - Detailed guideline about the models selected (how the work, how to train/adapt them, and how to use them).
    - Explicit description of the process followed during the benchmarking.

## Notes:
* Pending to add link to the published version of the paper.     

## Author
Erick Romero Kramer - [email](erickkramer@gmail.com) - [University email](erick.romero@smail.inf.h-brs.de)
