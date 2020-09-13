from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import neptune


def get_exp_params(exp_name, project_name):
    project = neptune.init(project_name)
    exp_list = project.get_experiments(exp_name)
    if len(exp_list) == 1:
        return exp_list[0].get_parameters()
    else:
        raise Exception("Wrong exp id!")
