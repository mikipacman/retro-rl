from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import neptune
import dill as pickle
import tempfile
import os


params_pickle_name = "params.pkl"


# Needs to be used inside 'with neptune.create_experiment()'
def save_exp_params(params):
    with tempfile.TemporaryDirectory(dir="/tmp") as temp:
        path_to_pickle = os.path.join(temp, params_pickle_name)
        pickle.dump(params, open(path_to_pickle, "wb"))
        neptune.send_artifact(path_to_pickle, params_pickle_name)


def get_exp_params(exp_name, project_name):
    project = neptune.init(project_name)
    exp_list = project.get_experiments(exp_name)
    if len(exp_list) == 1:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            exp_list[0].download_artifact(params_pickle_name, temp)
            return pickle.load(open(os.path.join(temp, params_pickle_name), "rb"))
    else:
        raise Exception("Wrong exp id!")
