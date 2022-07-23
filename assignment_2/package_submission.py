##############################
### FILL IN THE FOLLOWING: ###
##############################

U_ID = "u6284513"  # You need to specify!

# Do NOT add any other files or folders or change the paths of the following files
# Only change the name of your solution PDF file.

# MAKE SURE TO CHECK THE CREATED ZIP FILE HAS EVERYTHING IT SHOULD HAVE BEFORE SUBMITTING!!!

SUBMISSION_LIST = [
    # No need to submit theory question sheet `./assignment_1.pdf` or code framework `./framework/*`
    './COMP8600_Assignment2_u6284513_u7236045.pdf',  # Change to the PDF of your theory solutions!
    './contribution.py',  # Make sure to include this!
    './boframework/kernels.py',  # Make sure to include this!
    './boframework/gp.py',  # Make sure to include this!
    './boframework/acquisitions.py',  # Make sure to include this!
    './boframework/bayesopt.py',  # Make sure to include this!
    './bayesopt_implementation_viewer.ipynb',  # Make sure to include this!
]


##############################
### CHECKING AND PACKAGING ###
##############################

import os, zipfile

assert len(SUBMISSION_LIST) == 7, "Added extra files in, should only have the files initially listed above"
assert './contribution.py' in SUBMISSION_LIST, "No ./contribution.py in submission list"
assert './boframework/kernels.py' in SUBMISSION_LIST, "No ./kernels.py in submission list"
assert './boframework/gp.py' in SUBMISSION_LIST, "No ./gp.py in submission list"
assert './boframework/acquisitions.py' in SUBMISSION_LIST, "No ./acquisitions.py in submission list"
assert './boframework/bayesopt.py' in SUBMISSION_LIST, "No ./bayesopt.py in submission list"
assert './bayesopt_implementation_viewer.ipynb' in SUBMISSION_LIST, "No ./bayesopt_implementation_viewer.ipynb in submission list"
assert any('.pdf' in fname for fname in SUBMISSION_LIST), "No PDF solution file in submission list"

for s in SUBMISSION_LIST:
    assert os.path.exists(s), f'File {s} does not exist'

# Check length and typing of U_ID
assert(len(U_ID) == 8)
assert(type(U_ID) == str)

def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

with zipfile.ZipFile(f'{U_ID}_assignment_2.zip', 'w') as f:
    for s in SUBMISSION_LIST:
        if os.path.isdir(s):
            for s_file in get_all_file_paths(s):
                f.write(s_file)
        else:
            f.write(s)

