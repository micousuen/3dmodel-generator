'''
Created on Mar 5, 2018

@author: micou
'''
import re
import os
import json
import random
import datetime
import threading

class utils:
    """
    Some common tools to use. Like parsing sentences, write and read json file.
    """
    def _log_info_generator(self, logtype="Info"):
        return datetime.datetime.now().isoformat(sep=" ")+" [{0:<6}]{1:7}: ".format(os.getpid(), str(logtype))
    
    def info(self, details = "Information"):
        print(self._log_info_generator("Info"), details)
        
    def warn(self, details = "Warning"):
        print(self._log_info_generator("Warning"), details)

    def error(self, details = "Error will cause Exit of program"):
        print(self._log_info_generator("Error"), details)
        exit(1)
        
    def random_permutation(self, input_list):
        """
        return a copy of random permutation of a given list
        """
        result = [n for n in input_list]
        for n in range(len(result)):
            random_pos = random.randint(n, len(result)-1)
            result[n], result[random_pos] = result[random_pos], result[n]
        return result 
            
        
    def parse_line(self, text_line, parse_tool="re"):
        # Use regex to find out all words
        if parse_tool == "re":
            return list(re.findall(re.compile("\w+'*\w*"), text_line.lower())) # or to use \w+-*'*\w*
    
    def write_to_json(self, obj, json_filepath="./none.json"):
        with open(json_filepath, "w") as f:
            json.dump(obj, f, sort_keys=True, indent=4, separators=(',', ':'))
            
    def read_from_json(self, json_filepath="./none.json"):
        with open(json_filepath,"r") as f:
            temp_data = json.load(f)
        return temp_data
    
    def transform_list2chunks(self, input_list, chunk_total_num):
        '''turn list into chunks'''
        result = [[] for _ in range(chunk_total_num)]
        state = 0
        for i in input_list:
            result[state].append(i)
            state=(state+1)%chunk_total_num   
        return result  
    
    def boolean(self, string):
        if re.match("^[yY].*$|^[tT]rue$", string) != None:
            return True
        elif re.match("^[nN].*$|^[fF]alse$", string) != None:
            return False

    def stringcheck_acceptAll(self, string):
        return True

    def stringcheck_boolean(self, string):
        """
        Check input is a string represent a boolean
        """
        string = str(string) # because of 2.7 input problem
        if re.match("^[yY].*$|^[nN].*$|^[tT]rue$|^[fF]alse$", string) != None:
            return True
        else:
            return False
    
    def stringcheck_int(self, string):
        """
        Check input is a legal integer
        """
        string = str(string) # because of 2.7 input problem
        if re.match("^-?\d+$", string) != None:
            return True
        else:
            return False
        
    def stringcheck_float(self, string):
        """
        Check input is a legal float
        """
        string = str(string) # because of 2.7 input problem
        if re.match("^(-?\d+|-?.\d+|-?\d+\.\d*)$", string) != None:
            return True
        else:
            return False 
        
    def stringcheck_str(self, string):
        """
        Check input is non-empty string
        """
        string = str(string) # because of 2.7 input problem
        if string != "":
            return True
        else:
            return False
        
    def stringcheck_id(self, string):
        """
        Check input is legal id
        """
        string = str(string) # because of 2.7 input problem
        if re.match("^[a-zA-Z_][a-zA-Z0-9_]*$", string) != None:
            return True
        else:
            return False  
    
    stringcheck_func2type = {stringcheck_boolean: boolean, 
                             stringcheck_int: (lambda _, x: int(x)),
                             stringcheck_str: (lambda _, x: str(x)), 
                             stringcheck_float: (lambda _, x: float(x)), 
                             stringcheck_id: (lambda _, x: str(x)), 
                             stringcheck_acceptAll: (lambda _, x:x)
                            }
    
if __name__ == "__main__":
    test = utils()
    test.info("test")
    test.warn("test")
    test.error("test")
