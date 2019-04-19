import numpy as np

# load course IDs and trim out text label
courseIDs = np.genfromtxt("course_descriptions.txt", usecols=(0));
courseIDs = courseIDs[1:]

# load course descriptions and trim out text label
courseDescs = np.genfromtxt("course_descriptions.txt", dtype=None, usecols=(1));
courseDescs = courseDescs[1:]