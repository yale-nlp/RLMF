import os
from abc import ABC, abstractmethod

class Task(ABC):

    def __init__(self, args):

        self.dataset_name = args.dataset_name
        self.data_df = None 
        self.num_samples = args.num_samples
        self.random_seed = args.random_seed
        self.task_type = None
        os.environ['PYTHONPATH'] = '.'

    @abstractmethod
    def get_data_df(self):
        """
        Return data as a df. Limited to num_samples examples.
        
        DF columns:
        - input_args:   each sample (and its components) stored as a tuple
        - targets:      target outputs / gold answers
        - ...           any other metadata columns
        """
        pass 

    def set_data_df(self, data_df):
        """
        Set data_df to given dataframe.
        
        DF columns:
        - input_args:   each sample (and its components) stored as a tuple
        - targets:      target outputs / gold answers
        - ...           any other metadata columns
        """
        self.data_df = data_df 

    def get_num_samples(self):
        """
        Limit self.data_df to self.num_samples samples, if specified. 
        Otherwise, no changes to self.data_df are made.
        """

        # Limit to num_samples examples
        if self.num_samples:
            self.data_df = self.data_df.sample(
                    n=min(self.num_samples, self.data_df.shape[0]), 
                    random_state=self.random_seed, 
                    replace=False,
                ).reset_index(drop=True)
        
    @abstractmethod
    def score(self, predictions):
        """
        Return dict of task performance metric(s) AND list of errors to add to results_df (if applicable, otherwise return list of 0's).
        """
        pass 