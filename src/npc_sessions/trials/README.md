- Dynamic Routing and Templeton experiments use
  https://github.com/samgale/DynamicRoutingTask/blob/main/TaskControl.py to
  coordinate stimuli, process subject responses and save parameters and trial data to disk
- data are saved as `.hdf5` 
- each subclass of `TaskControl` customizes the experiment, and can add new fields
  to the resulting `.hdf5` file

This module deals with creating a trials dataframe for each experiment, for adding
to an NWB file. 

Although each NWB file ususally has a main `trials` table, it allows for extra tables
to be added to `intervals` (a dict mapping `name: table`). We create a few tables
each session, for RF mapping, optotagging, and the main behavior task 


-  each module in `TaskControl` contains one class, mirroring the subclasses of
   `TaskControl` in https://github.com/samgale/DynamicRoutingTask

    - each class reads an `.hdf5` file and interprets the contents
    
    - for `DynamicRouting1` (the main behavior task) a lot of the work is already done
    by
    [`DynRoutData`](https://github.com/samgale/DynamicRoutingTask/blob/main/Analysis/DynamicRoutingAnalysisUtils.py),
    which we import and use

  - each class is a subclass of `PropertyDict`, which just provides a convenient
    way to export the values of all property getters that have a name with no leading
    underscore
    - each property is an iterable of values (times, parameters, bools, etc.),
      with one value per trial
    - each property (and its name) becomes a column name in the trials table
        - the property's docstring also becomes the column description 
    - all properties must therefore be equal in length 

## naming conventions and types
- `..._time` : `float` - event times in seconds, missing values as nan

- `..._index` / `index_...` : `float` - integer values, missing values as nan

- `is_...` : `bool` - missing or not applicable values as `False`
    - intended as a positive mask for filtering trials:

        ```python
        trials.query('is_hit').query('is_vis_rewarded')
        ```