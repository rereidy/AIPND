#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Ron Reidy
# DATE CREATED:
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import sys
from pathlib import Path

import argparse
from time import time, sleep
from os import listdir
from os.path import join
from os.path import isdir
import re

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# because these are not importable from classifier.py
ARCH = ['vgg', 'resnet', 'alexnet']

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()
    
    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    if in_arg.dbg == 1:
        print("command line args", file=sys.stderr)
        for arg in vars(in_arg):
            print("\t{0:<7} {1}".format(arg, getattr(in_arg, arg)), file=sys.stderr)
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)
    if in_arg.dbg == 1:
        print("answers_dic len={0:d}".format(len(answers_dic)))
        print("answers_dic=", answers_dic)

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    if in_arg.dbg == 1:
        print("result_dic len={0:d}".format(len(result_dic)))
        mtch, nomtch = 0, 0
        for k, v in result_dic.items():
            if v[2] > 0:
                mtch += 1
                print("MATCH: ", end="")
            else:
                nomtch += 1
                print("NO MATCH: ", end="")
            print(k, v)
        print("total matches={0:d}".format(mtch))
        print("total non-match={0:d}".format(nomtch))
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)
    if in_arg.dbg == 1:
        for k, v in result_dic.items():
            print("{0:<30s}: [{1}, {2}, {3},".format(k, v[0], v[1], v[2]), end="")
            if len(v) == 5:
                if v[3] == 1:
                    print(" is labeled a dog,", end="")
                else:
                    print(" is NOT labeled a dog,", end="")
                
                if v[4] == 1:
                    print(" is classified as a dog]")
                else:
                    print(" is NOT classified as a dog]")
            else:
                print("error in adjust_results4_isadog(): invalid array count: {0:d}: '{1}'".format(len(v), v))

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)
    if in_arg.dbg == 1:
        print("results")
        for k, v in results_stats_dic.items():
            print("{0:<25}: ".format(k), end="")
            if isinstance(v, int):
                print("{0:3d}".format(v))
            elif isinstance(v, float):
                print("{0:5.2f}".format(v))

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, int(in_arg.print_mismatch == 1), int(in_arg.print_mismatch == 1))

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime: {0:02d}:{1:02d}:{2:02d}".format(int(tot_time / 3600), 
                                                                       int((tot_time%3600) / 60), 
                                                                       int((tot_time%3600) % 60)
                                                                      )
         )

# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type    = str,
                        default = 'pet_images/',
                        help    = 'path to the folder pet_images'
                       )
    
    parser.add_argument('--arch',
                        type    = str,
                        default = "vgg",
                        help    = 'chosen CNN model'
                       )
    
    parser.add_argument('--dogfile',
                        type    = str,
                        default = 'dognames.txt',
                        help    = 'name of file with dog names'
                       )
    
    parser.add_argument('--dbg',
                        type    = int,
                        default = 1,
                        help    = 'turn debug on/off [default on]'
                       )
    
    parser.add_argument('--print_mismatch',
                        type    = int,
                        default = 1,
                        help    = 'print mismatch dog classifications [default on]'
                       )
    
    args = parser.parse_args()
    
    images = Path(args.dir)
    if not images.exists() and not images.is_dir():
        raise ValueError("invalid dir argument: directory '{0}' does not exist".format(args.dir))
        
    df = Path(args.dogfile)
    if not df.exists():
      	raise ValueError("invalid dogfile argument: file '{0}' does not exist".format(args.dogfile))
        
    if args.arch not in ARCH:
        raise ValueError("invalid arch argument: arch '{0}' must be one of '{1}'".format(args.arch, ",".join(ARCH)))
        
    return args
  
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    petlabels_dic = dict()
    for fname in listdir(image_dir):
        if isdir(join(image_dir, fname)):
            print("directory encountered: '{0}' - skipped".format(join(image_dir, fname)))
            continue

        if fname not in petlabels_dic:
	        petlabels_dic[fname] = make_pet_label(fname)
        else:
            print("duplicate file name: {0}".format(fname))
    
    return petlabels_dic
  
def make_pet_label(label_string):
    """
    make_pet_label - create teh pet label from the file name
    
    Parameters:
        label_string - the file name
        
    Return:
        the words of the file name (words delimited by '_')
    """
    plabel = ""
    for word in label_string.lower().split("_"):
        if word.isalpha():
            plabel += word + " "

    return plabel.strip()

def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()
    for fname in petlabel_dic.keys():
        """
        watch for the race condition of files being identified in the petlabel_dic data 
        structure that may have been deleted/renamed before executing this function
        """
        full_path = Path(join(images_dir, fname))
        if full_path.exists():
            image_classification = classifier(str(full_path), model).lower().strip()
            found = classification_match(petlabel_dic[fname], re.split(r",\s+", image_classification))
            results_dic[str(full_path)] = [petlabel_dic[fname], image_classification, found]
        else:
            print("file does not exist: {0}".format(full_path))
    
    return results_dic
  
def classification_match(real_name, class_array):
    """
    classification_match - string match of real pet label and immage classification of pet (CNN)
    
    Parameters:
            real_name - petlebel name derived from file name
            class_array - array of strings returned from classifier() function
     Return:
            found - value of 1 - match found; value of 0 - no match found
    """
    found = 0
    for name in class_array:
        sfnd = name.find(real_name)
        if sfnd >= 0:
            if name[sfnd:] == real_name:
                found += 1
            
        if found != 0:
            break
            
    return found

def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dognames = dict()
    lineno = 0
    with open(dogfile, "r") as f:
        for line in f.readlines():
            lineno += 1
            if line.startswith("\n"):
                continue
                
            ltext = line.rsplit("\n")[0].lower()
            if ltext not in dognames:
                dognames[ltext] = lineno
            else:
                print("duplicate dogname found: {)}".format(ltext))
    
    for k, v in results_dic.items():
        pet_image_label_dog, classification_dog = is_dog(dognames, v[0], v[1])
        v.append(pet_image_label_dog)
        v.append(classification_dog)

def is_dog(dognames, label_str, classification_str):
    """
    is_dog = look up names in dognames dictionary
    
    Parameters:
        dognames - dictionary of dognames from dognames.txt
        label_str - the pet labelfrom the file name
        classification_str - the classification names from classifier()
    """
    
    pet_label, class_label = 0, 0
    
    if label_str in dognames:
        pet_label += 1
        
    for word in re.split(r",\s+", classification_str):
        if word in dognames:
            class_label += 1
            break
    
    return pet_label, class_label

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    results_stats = dict()

    results_stats['cnt_image']           = len(results_dic)
    results_stats['cnt_dogs_image']      = 0
    results_stats['cnt_match']           = 0
    results_stats['cnt_correct_dogs']    = 0
    results_stats['cnt_correct_notdogs'] = 0
    results_stats['cnt_correct_breed']   = 0
    
    for val in results_dic.values():
        if val[2] == 1:
            results_stats['cnt_match'] += 1
            
        if sum(val[2:]) == 3:
            results_stats['cnt_correct_breed'] += 1
            
        if val[3] == 1:
            results_stats['cnt_dogs_image'] += 1
            
            if val[4] == 1:
                results_stats['cnt_correct_dogs'] += 1
        else:
            results_stats['cnt_correct_notdogs'] += 1

    results_stats['cnt_notdogs_img']   = results_stats['cnt_image'] - results_stats['cnt_dogs_image']
    results_stats['pct_match']         = (results_stats['cnt_match']/results_stats['cnt_image'])*100
    results_stats['pct_correct_dogs']  = (results_stats['cnt_correct_dogs']/results_stats['cnt_dogs_image'])*100
    results_stats['pct_correct_breed'] = (results_stats['cnt_correct_breed']/results_stats['cnt_dogs_image'])*100
    try:
        results_stats['pct_correct_not_dogs'] = (results_stats['cnt_correct_notdogs']/results_stats['cnt_notdogs_img'])*100
    except ZeroDivisionError:
        results_stats['pct_correct_not_dogs'] = 0.0

    return results_stats

def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    print("CNN arch model: {0}".format(model))
    print("Number of images total: {0:d}".format(results_stats['cnt_image']))
    print("number of dog images: {0:d}".format(results_stats['cnt_dogs_image']))
    print("number of non-dog images: {0:d}".format(results_stats['cnt_correct_notdogs']))
    
    for k, v in results_stats.items():
        if k.startswith("pct_"):
            print("{0:<25}: ".format(k), end="")
            print("{0:5.2f}".format(v))
    
    if print_incorrect_dogs:
        if results_stats['cnt_correct_dogs'] + results_stats['cnt_correct_notdogs'] != results_stats['cnt_image']:
            print("\nIncorrect dog counts found")
            for k in results_dic.keys():
                if sum(results_dic[k][3:]) == 1:
                    print("Actual: {0:<30}; Classification: {1:<30}".format(results_dic[k][0], results_dic[k][1]))
      
    if print_incorrect_breed:
        if results_stats['cnt_correct_dogs'] != results_stats['cnt_correct_breed']:
            print("\nIncorrect dog dog breed assigmnets found")
            for k in results_dic.keys():
                if sum(results_dic[k][3:]) == 2 and results_dic[k][2] == 0:
                    print("Actual: {0:<30}; Classification: {1:<30}".format(results_dic[k][0], results_dic[k][1]))

# Call to main function to run the program
if __name__ == "__main__":
    rc = 0
    try:
        main()
    except ValueError as e:
        print(e, file=sys.stderr)
        rc += 1
    except Exception as e:
        print(e, file=sys.stderr)
        rc += 1
    finally:
        sys.exit(rc)
